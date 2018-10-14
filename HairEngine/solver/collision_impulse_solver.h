#pragma once

#include <utility>
#include <vector>
#include <cstdio>

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
		 * @param checkingDistance The distance to check whether it should be further tested by collision engine
		 * @param kCollision The collision spring stiffness
		 * @param maxCollisionPerSegment The max collision count for a edge (line segment)
		 * @param maxCollisionForceCount In some situtaions, a sudden impulse force makes the system vibrated, 
		 * we limit the maximum collision force by diving it if the total collision spring excceds this value
		 */
		CollisionImpulseSolver(float checkingDistance, int maxCollisionPerSegment, float kCollision, int maxCollisionForceCount = 4):
			segmentKnnSolver(nullptr),
			checkingDistance(checkingDistance),
			kCollision(kCollision), 
			maxCollisionPerSegment(maxCollisionPerSegment),
			maxCollisionForceCount(maxCollisionForceCount) {}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			segmentKnnSolver = new SegmentKNNSolver(checkingDistance);
			segmentKnnSolver->setup(hair, currentTransform);

			velocityImpulses.resize(hair.nparticle);
			numVelocityImpulses.resize(hair.nparticle);
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {

			std::cout << "[CollisionImpulseSolver]: Building knn..." << std::endl;

			segmentKnnSolver->solve(hair, info);

			std::cout << "[CollisionImpulseSolver]: Get collisions..." << std::endl;

			// Use iteration
			for (int i = 0; i < 5; ++i) {
				_solve(hair, info);
			}
		}

		void tearDown() override {
			segmentKnnSolver->tearDown();
			delete segmentKnnSolver;
		}

	HairEngine_Protected:

		void _solve(Hair& hair, const IntegrationInfo& info) {

//			std::fill(velocityImpulses.begin(), velocityImpulses.end(), Eigen::Vector3f::Zero());
//			std::fill(numVelocityImpulses.begin(), numVelocityImpulses.end(), 0);

			for (int idx1 = 0; idx1 < hair.nsegment; ++idx1) {

				auto seg1 = hair.segments + idx1;

				const int numNeighbor = std::max(maxCollisionPerSegment, segmentKnnSolver->getNNeighbourForSegment(idx1));
				for (int i = 0; i < numNeighbor; ++i) {
					const int idx2 = segmentKnnSolver->getNeighbourIndexForSegment(idx1, i);
					auto seg2 = hair.segments + idx2;

					if (idx2 <= idx1)
						continue;

					std::pair<float, float> ts = MathUtility::linetoLineDistanceClosestPointApproach(seg1->p1->pos, seg1->p2->pos, seg2->p1->pos, seg2->p2->pos);

					if (ts.first > 0.0f && ts.first < 1.0 && ts.second > 0.0f && ts.second < 1.0f) {

						Eigen::Vector3f dn = d(seg1, seg2, ts.first, ts.second);

						const float l0 = dn.norm();
						dn /= l0;

//						auto s1v1 = getVelocity(seg1->p1->globalIndex);
//						auto s1v2 = getVelocity(seg1->p2->globalIndex);
//						auto s2v1 = getVelocity(seg2->p1->globalIndex);
//						auto s2v2 = getVelocity(seg2->p2->globalIndex);
						auto &s1v1 = seg1->p1->vel;
						auto &s1v2 = seg1->p2->vel;
						auto &s2v1 = seg2->p1->vel;
						auto &s2v2 = seg2->p2->vel;

						Eigen::Vector3f s2v = MathUtility::lerp(seg2->p1->vel, seg2->p2->vel, ts.second);
						Eigen::Vector3f s2nv = MathUtility::project(s2v, dn);

						Eigen::Vector3f s1v = MathUtility::lerp(seg1->p1->vel, seg1->p2->vel, ts.first);
						Eigen::Vector3f s1nv = MathUtility::project(s1v, dn);

						Eigen::Vector3f dnv = s2nv - s1nv;

//						printf("Before, dn: {%f, %f, %f}, l0: %f\n", dn.x(), dn.y(), dn.z(), l0);
//						printf("Before, seg1(%d) --> v1: {%f, %f, %f}, v2: {%f, %f, %f}, pv: {%f, %f, %f}, nv: {%f, %f, %f}\n", idx1, s1v1.x(), s1v1.y(), s1v1.z(), s1v2.x(), s1v2.y(), s1v2.z(), s1v.x(), s1v.y(), s1v.z(), s1nv.x(), s1nv.y(), s1nv.z());
//						printf("Before, seg2(%d) --> v1: {%f, %f, %f}, v2: {%f, %f, %f}, pv: {%f, %f, %f}, nv: {%f, %f, %f}\n", idx2, s2v1.x(), s2v1.y(), s2v1.z(), s2v2.x(), s2v2.y(), s2v2.z(), s2v.x(), s2v.y(), s2v.z(), s2nv.x(), s2nv.y(), s2nv.z());

						// If it not seems to collide in current simulation frame, just continue
						if (dnv.dot(dn) >= 0.0f || dnv.norm() * info.t < l0)
							continue;

						// If true collide, compute the impulse force, we assume
						Eigen::Vector3f v = (s2v + s1v) / 2.0f;

//						printf("v: {%f, %f, %f}\n", v.x(), v.y(), v.z());

						Eigen::Vector3f dv = v - s1v;

//						addVelociyImpulse(seg1->p1->globalIndex, dv);
//						addVelociyImpulse(seg1->p2->globalIndex, dv);
//						addVelociyImpulse(seg2->p1->globalIndex, -dv);
//						addVelociyImpulse(seg2->p2->globalIndex, -dv);

						s1v1 += dv;
						s1v2 += dv;
						s2v1 -= dv;
						s2v2 -= dv;

//						printf("After, seg1 --> v1: {%f, %f, %f}, v2: {%f, %f, %f}\n", s1v1.x(), s1v1.y(), s1v1.z(), s1v2.x(), s1v2.y(), s1v2.z());
//						printf("After, seg2 --> v1: {%f, %f, %f}, v2: {%f, %f, %f}\n", s2v1.x(), s2v1.y(), s2v1.z(), s2v2.x(), s2v2.y(), s2v2.z());
					}
				}
			}
		}

	HairEngine_Protected:

		float checkingDistance; ///< The distance to check whether it should be further test with collision engine
		float kCollision; ///< The collision spring stiffness
		int maxCollisionPerSegment; ///< The max collision count for an edge (line segment)

		/// Too many collision force will makes the system vibrated, we limit the total force for a segment 
		/// by diving it if it the total force count larger than the "maxCollisionForceCount"
		int maxCollisionForceCount;

		/// A pre-allocated space for storing all the collision information, since we 
		/// limit the size of max collision / segment, all the collision info for segment i is stored in the 
		/// [ collisionInfos[i * maxCollisionPerSegment], collisionInfos[(i + 1) * maxCollisionPerSegment ).
		SegmentKNNSolver *segmentKnnSolver;

		std::vector<Eigen::Vector3f> velocityImpulses;
		std::vector<int> numVelocityImpulses;

		Eigen::Vector3f d(Hair::Segment::Ptr seg1, Hair::Segment::Ptr seg2, float t1, float t2) {
			return seg2->lerpPos(t2) - seg1->lerpPos(t1);
		}

		Eigen::Vector3f dv(Hair::Segment::Ptr seg1, Hair::Segment::Ptr seg2, float t1, float t2) {
			return seg2->lerpVel(t2) - seg1->lerpVel(t1);
		}

		Eigen::Vector3f predicitedD(Hair::Segment::Ptr seg1, Hair::Segment::Ptr seg2, float t1, float t2, float time) {
			return d(seg1, seg2, t1, t2)  + time * dv(seg1, seg2, t1, t2);
		}

		Eigen::Vector3f getVelocity(int i) const {
			float scale = (numVelocityImpulses[i] > maxCollisionForceCount) ? static_cast<float>(maxCollisionForceCount) / numVelocityImpulses[i] : 1.0f;
			return hair->particles[i].vel + scale * velocityImpulses[i];
		}

		void addVelociyImpulse(int i, const Eigen::Vector3f & dv) {
			++numVelocityImpulses[i];
			velocityImpulses[i] += dv;
		}
	};
}
