#include <vector>
#include <algorithm>
#include "segment_knn_solver.h"
#include "HairEngine/HairEngine/util/mathutil.h"

namespace HairEngine {
	/**
	 * The solver that resolve hair contacts by using impulse based spring forces 
	 * in semi implicit euler. It will add the force to the "im" property of the particles.
	 * The HairContactsImpulseSolver needs the SegmentKNNSolver for finding the k nearest neighbour 
	 * and add forces between them.
	 */
	class HairContactsImpulseSolver: public Solver {
	HairEngine_Public:
		/**
		 * Constructor
		 * 
		 * @param segmentKnnSolver The SegmentKNNSolver used for neighbour search, added it before this solver to 
		 * build the knn acceleration structure.
		 * @param kContactSpring The stiffness of the contact 
		 */
		HairContactsImpulseSolver(SegmentKNNSolver *segmentKnnSolver,float kContactSpring): 
			segmentKnnSolver(segmentKnnSolver), kContactSpring(kContactSpring){}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			// Setup the usedBuffers, assign a individual used buffer for each thread to avoid race condition
			for (int i = 0; i < ParallismUtility::getOpenMPMaxHardwareConcurrency(); ++i) {
				usedBuffers.emplace_back(hair.nparticle, -1);
			}

			// Setup the contact springs
			for (int i = 0; i < hair.nparticle; ++i)
				contactSprings.emplace_back();
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {
			const auto r = segmentKnnSolver->getRadius();
			const auto r2 = r * r;

			// Erase all the distance larger the r
			ParallismUtility::parallelFor(0, hair.nsegment, [this, &hair, r2](int idx1) {
				auto & cs = contactSprings[idx1];
				auto seg1 = hair.segments + idx1;

				const auto removeEnd = std::remove_if(cs.begin(), cs.end(), [seg1, r2](const ContactSpringInfo & _) -> bool {
					return (seg1->midpoint() - (hair.segments + _.idx2)->midpoint()).squaredNorm() > r2;
				});
				cs.erase(removeEnd, cs.end());
			});

			// Sum up the force, since the newly created spring yield 0 force
			// TODO: Add parallism
			for (int idx1 = 0; idx1 < hair.nsegment; ++idx1) {
				for (const auto & _ : contactSprings[idx1]) {
					const Eigen::Vector3f springForce = MathUtility::massSpringForce4f((hair.segments + idx1)->midpoint(), 
						(hair.segments + _.idx2)->midpoint(), kContactSpring, _.l0).segment<3>(0);

					(hair.particles + idx1)->impulse += springForce;
					(hair.particles + _.idx2)->impulse -= springForce;
				}
			}

			// Add additional
			ParallismUtility::parallelForWithThreadIndex(0, hair.nsegment, [this, &hair](int idx1, int threadID) {
				auto & usedBuffer = usedBuffers[threadID];
				auto & cs = contactSprings[idx1]; 
				const auto seg1 = hair.segments + idx1;

				// Add others to the used buffer
				for (const auto & _ : cs)
					usedBuffer[_.idx2] = idx1;

				const auto nConnection = segmentKnnSolver->getNNeighbourForSegment(idx1);
				for (int i = 0; i < nConnection; ++i) {
					const auto idx2 = segmentKnnSolver->getNeighbourIndexForSegment(idx1, i);

					// To remove duplicate adding for pair (idx1, idx2) and ensure the spring is not created previously
					if (idx1 < idx2 && usedBuffer[idx2] != idx1) {
						const auto seg2 = hair.segments + idx2;
						cs.emplace_back(idx2, (seg1->midpoint() - seg2->midpoint()).norm());
					}
				}
			});
		}

	HairEngine_Protected :

		struct ContactSpringInfo {
			int idx2; ///< The index for another endpoint
			float l0; ///< The rest length when creating

			ContactSpringInfo(int idx2, float l0): idx2(idx2), l0(l0) {}
		};

		SegmentKNNSolver *segmentKnnSolver;
		float kContactSpring; ///< The stiffness of the contact spring
		std::vector<std::vector<ContactSpringInfo>> contactSprings; ///< Index array of the contacts spring
		std::vector<std::vector<int>> usedBuffers; ///< Used in iteration to indicate whether the spring has been created
	};
}
