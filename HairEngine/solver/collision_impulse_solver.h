#pragma once

#include <utility>
#include <vector>

#include "integration_info.h"
#include "segment_knn_solver.h"
#include "HairEngine/HairEngine/util/mathutil.h"

namespace HairEngine {
	class CollisionImpulseSolver: public Solver {
	HairEngine_Public:

		/**
		 * Constructor
		 * 
		 * @param segmentKnnSolver The segment k-nearest solver 
		 * @param kCollision The collision spring stiffness
		 * @param maxCollisionPerSegment The max collision count for a edge (line segment)
		 * @param maxCollisionForceCount In some situtaions, a sudden impulse force makes the system vibrated, 
		 * we limit the maximum collision force by diving it if the total collision spring excceds this value
		 */
		CollisionImpulseSolver(SegmentKNNSolver *segmentKnnSolver, float kCollision, int maxCollisionPerSegment, int maxCollisionForceCount = 4): 
			segmentKnnSolver(segmentKnnSolver), kCollision(kCollision), maxCollisionPerSegment(maxCollisionPerSegment), maxCollisionForceCount(maxCollisionForceCount) {}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			HairEngine_AllocatorAllocate(collisionInfos, hair.nsegment * maxCollisionPerSegment);
			ncollision = std::vector<int>(hair.nsegment, 0);
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {
			// Remove invalid collision springs
			ParallismUtility::parallelForWithThreadIndex(0, hair.nsegment, [this, &hair, &info] (int idx1, int threadId) {

				const auto range = getCollisionRange(idx1);
				auto seg1 = hair.segments + idx1;

				const auto & removePredicate = [&hair, &info, seg1](const CollisionInfo & _) -> bool {
					const auto seg2 = hair.segments + _.idx2;

					const Eigen::Vector3f d = this->predicitedD(seg1, seg2, _.t1, _.t2, info.t);

					// The collision will be removed if it is in the predicted direction is in the same direction and larger than rest length
					return d.dot(_.dn) > 0.0f && d.squaredNorm() > _.l0 * _.l0;
				};

				const auto removeEnd = std::remove_if(range.first, range.second, removePredicate);

				// Update the ncollision
				ncollision[idx1] = static_cast<int>(removeEnd - range.first);

				// Compute the collision force
				Eigen::Vector3f force;
				for (int i = 0; i < ncollision[idx1]; ++i) {
					const auto _ = range.first + i;
					auto seg2 = hair.segments + _->idx2;

					force += MathUtility::massSpringForce(
						seg1->lerpPos(_->t1),
						seg2->lerpPos(_->t2),
						kCollision, _->l0
					);
				}
				if (ncollision[idx1] > maxCollisionForceCount) {
					force *= static_cast<float>(maxCollisionForceCount) / static_cast<float>(ncollision[idx1]);
				}

				syncLock.lock();
				seg1->p1->impulse += force;
				seg1->p2->impulse += force;
				syncLock.unlock();

				// Add other collision springs
				const int nneeds = std::min(segmentKnnSolver->getNNeighbourForSegment(idx1), maxCollisionPerSegment - ncollision[idx1]);
				for (int i = 0; i < nneeds; ++i) {
					const int idx2 = segmentKnnSolver->getNeighbourIndexForSegment(idx1, i);
					const auto seg2 = hair.segments + idx2;

					auto ts = MathUtility::linetoLineDistanceClosestPointApproach(seg1->p1->pos, seg1->p2->pos, seg2->p1->pos, seg2->p2->pos);
					if (ts.first > 0.0f && ts.first < 1.0 && ts.second > 0.0f && ts.second < 1.0f) {

						Eigen::Vector3f dn = d(seg1, seg2, ts.first, ts.second);
						Eigen::Vector3f dnv = dv(seg1, seg2, ts.first, ts.second);

						const float l0 = dn.norm();
						dn /= l0;

						// If it not seems to collide in current simulation frame, just continue
						if (dnv.dot(dn) >= 0.0f || dnv.norm() * info.t < l0)
							continue;

						//Collision happen and added
						std::allocator<CollisionInfo>().construct(range.first + (ncollision[idx1]++), idx2, ts.first, ts.second, l0);
					}
				}
			});
		}

		void tearDown() override {
			HairEngine_AllocatorDeallocate(collisionInfos, hair->nsegment * maxCollisionPerSegment);
		}

	HairEngine_Protected:

		struct CollisionInfo {
			int idx2; ///< Another collision segment index
			float t1, t2; ///< The CPA (cloest point approach) of the two line segments
			Eigen::Vector3f dn; ///< The normalized direction for detecting the collision
			float l0; ///< The rest length of the collision setup

			CollisionInfo(int idx2, float t1, float t2, const Eigen::Vector3f & dn, float l0):
				idx2(idx2), t1(t1), t2(t2), dn(dn), l0(l0) {}
		};

		float kCollision; ///< The collision spring stiffness
		int maxCollisionPerSegment; ///< The max collision count for an edge (line segment)

		/// Too many collision force will makes the system vibrated, we limit the total force for a segment 
		/// by diving it if it the total force count larger than the "maxCollisionForceCount"
		int maxCollisionForceCount; 

		/// A pre-allocated space for storing all the collision information, since we 
		/// limit the size of max collision / segment, all the collision info for segment i is stored in the 
		/// [ collisionInfos[i * maxCollisionPerSegment], collisionInfos[(i + 1) * maxCollisionPerSegment ).
		SegmentKNNSolver *segmentKnnSolver;
		CollisionInfo *collisionInfos = nullptr;
		std::vector<int> ncollision; ///< Number of collision for segment i

		CompactNSearch::Spinlock syncLock; ///< Use to protect the modification of particle impulse

		/**
		 * Helper function, get the valid collision information array range for segment i
		 * 
		 * @param segmentIndex The segment index
		 * @return A pointer range specifying the range [start, end) for the valid collision information array for segment i
		 */
		std::pair<CollisionInfo *, CollisionInfo *> getCollisionRange(int segmentIndex) const {
			std::pair<CollisionInfo *, CollisionInfo *> ret;
			ret.first = collisionInfos + segmentIndex * maxCollisionPerSegment;
			ret.second = ret.first + ncollision[segmentIndex];

			return ret;
		}

		Eigen::Vector3f d(Hair::Segment::Ptr seg1, Hair::Segment::Ptr seg2, float t1, float t2) {
			return seg2->lerpPos(t2) - seg1->lerpPos(t1);
		}

		Eigen::Vector3f dv(Hair::Segment::Ptr seg1, Hair::Segment::Ptr seg2, float t1, float t2) {
			return MathUtility::lerp(seg2->p1->vel, seg2->p2->vel, t2) - MathUtility::lerp(seg1->p1->vel, seg1->p2->vel, t1);
		}

		Eigen::Vector3f predicitedD(Hair::Segment::Ptr seg1, Hair::Segment::Ptr seg2, float t1, float t2, float time) {
			return d(seg1, seg2, t1, t2)  + time * dv(seg1, seg2, t1, t2);
		}
	};
}
