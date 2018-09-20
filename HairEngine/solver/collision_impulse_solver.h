#pragma once

#include <utility>
#include <vector>

#include "integration_info.h"
#include "segment_knn_solver.h"
#include "../util/mathutil.h"

namespace HairEngine {
	class CollisionImpulseSolver: public Solver {

		friend class HairContactsAndCollisionImpulseSolverVisualizer;

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
		CollisionImpulseSolver(SegmentKNNSolver *segmentKnnSolver, int maxCollisionPerSegment, float kCollision, int maxCollisionForceCount = 4):
			segmentKnnSolver(segmentKnnSolver), 
			kCollision(kCollision), 
			maxCollisionPerSegment(maxCollisionPerSegment),
			maxCollisionForceCount(maxCollisionForceCount) {}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			HairEngine_AllocatorAllocate(collisionInfos, hair.nsegment * maxCollisionPerSegment);
			ncollision = std::vector<int>(hair.nsegment, 0);
			forces = std::vector<ForceBuffer>(hair.nparticle);

			for (int i = 0; i < ParallismUtility::getOpenMPMaxHardwareConcurrency(); ++i)
				usedBuffers.emplace_back(hair.nsegment, -1);
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {

			for (int i = 0; i < ParallismUtility::getOpenMPMaxHardwareConcurrency(); ++i) {
				std::fill(usedBuffers[i].begin(), usedBuffers[i].end(), -1);
			}

			std::fill(forces.begin(), forces.end(), ForceBuffer());

			ParallismUtility::parallelForWithThreadIndex(0, hair.nsegment, [this, &hair, &info] (int idx1, int threadId) {

				const auto range = getCollisionRange(idx1);
				auto seg1 = hair.segments + idx1;
				auto &usedBuffer = usedBuffers[threadId];

				// Remove invalid collision springs
				const auto & removePredicate = [this, &hair, &info, seg1](const CollisionInfo & _) -> bool {
					const auto seg2 = hair.segments + _.idx2;

					const Eigen::Vector3f predictedD = this->predicitedD(seg1, seg2, _.t1, _.t2, info.t);

					// The collision will be removed if it is in the predicted direction is in the same direction and larger than rest length
					return predictedD.dot(_.dn) >= 0.0f;
				};

				const auto removeEnd = std::remove_if(range.first, range.second, removePredicate);

				// Update the ncollision
				ncollision[idx1] = static_cast<int>(removeEnd - range.first);

				// Compute the collision force
				for (int i = 0; i < ncollision[idx1]; ++i) {

					usedBuffer[i] = idx1;

					const auto _ = range.first + i;
					auto seg2 = hair.segments + _->idx2;

					Eigen::Vector3f force = MathUtility::massSpringForce(seg1->lerpPos(_->t1), seg2->lerpPos(_->t2), kCollision, _->l0);

					syncLock.lock();
					forces[seg1->p1->globalIndex] += force;
					forces[seg1->p2->globalIndex] += force;
					forces[seg2->p1->globalIndex] -= force;
					forces[seg2->p2->globalIndex] -= force;
					syncLock.unlock();
				}

				// Add other collision springs
				const int nneeds = std::min(segmentKnnSolver->getNNeighbourForSegment(idx1), maxCollisionPerSegment - ncollision[idx1]);
				for (int i = 0; i < nneeds; ++i) {
					const int idx2 = segmentKnnSolver->getNeighbourIndexForSegment(idx1, i);

					if (idx2 <= idx1 || usedBuffer[idx2] == idx1)
						continue;

					const auto seg2 = hair.segments + idx2;

					std::pair<float, float> ts;
					MathUtility::lineSegmentSquaredDistance(seg1->p1->pos, seg1->p2->pos, seg2->p1->pos, seg2->p2->pos, ts.first, ts.second);
					if (ts.first > 0.0f && ts.first < 1.0 && ts.second > 0.0f && ts.second < 1.0f) {

						Eigen::Vector3f dn = d(seg1, seg2, ts.first, ts.second);

						const float l0 = dn.norm();
						dn /= l0;

						Eigen::Vector3f s2nv = MathUtility::lerp(seg2->p1->vel, seg2->p2->vel, ts.second);
						s2nv = MathUtility::project(s2nv, dn);
						Eigen::Vector3f s1nv = MathUtility::lerp(seg1->p1->vel, seg1->p2->vel, ts.first);
						s1nv = MathUtility::project(s1nv, dn);

						Eigen::Vector3f dnv = s2nv - s1nv;

						// If it not seems to collide in current simulation frame, just continue
						if (dnv.dot(dn) >= 0.0f || dnv.norm() * info.t < l0)
							continue;

						//Collision happen and added
						std::allocator<CollisionInfo>().construct(range.first + (ncollision[idx1]++), idx2, ts.first, ts.second, dn, l0);
					}
				}
			});

			// Commit the force to particles
			mapParticle(true, [this] (Hair::Particle::Ptr par) {
				const auto & f = forces[par->globalIndex];
				const float scale = (f.count > maxCollisionForceCount) ? static_cast<float>(maxCollisionForceCount) / f.count : 1.0f;
				par->impulse += f.force * scale;
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

		struct ForceBuffer {
			int count;
			Eigen::Vector3f force;

			ForceBuffer(): count(0), force(Eigen::Vector3f::Zero()) {}

			ForceBuffer &operator+=(const Eigen::Vector3f & f) {
				++count;
				force += f;
				return *this;
			}

			ForceBuffer &operator-=(const Eigen::Vector3f & f) {
				++count;
				force -= f;
				return *this;
			}
		};

		std::vector<ForceBuffer> forces;

		/// A pre-allocated space for storing all the collision information, since we 
		/// limit the size of max collision / segment, all the collision info for segment i is stored in the 
		/// [ collisionInfos[i * maxCollisionPerSegment], collisionInfos[(i + 1) * maxCollisionPerSegment ).
		SegmentKNNSolver *segmentKnnSolver;
		CollisionInfo *collisionInfos = nullptr;
		std::vector<int> ncollision; ///< Number of collision for segment i

		std::vector<std::vector<int>> usedBuffers; ///< The buffer to check whether the collision springs have been inserted

		//FIXME
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
			return seg2->lerpVel(t2) - seg1->lerpVel(t1);
		}

		Eigen::Vector3f predicitedD(Hair::Segment::Ptr seg1, Hair::Segment::Ptr seg2, float t1, float t2, float time) {
			return d(seg1, seg2, t1, t2)  + time * dv(seg1, seg2, t1, t2);
		}
	};
}
