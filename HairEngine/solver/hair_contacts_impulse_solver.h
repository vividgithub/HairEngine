#pragma once

#include <vector>
#include <algorithm>
#include <iostream>
#include <queue>
#include <utility>

#include "CompactNSearch.h"
#include "VPly/vply.h"
#include "../util/mathutil.h"
#include "../old/finite_grid.h"

#include "visualizer.h"
#include "segment_knn_solver.h"
#include "hair_contacts_and_collision_impulse_visualizer.h"

namespace HairEngine {

	/**
	 * The solver that resolve hair contacts by using impulse based spring forces 
	 * in semi implicit euler. It will add the force to the "im" property of the particles.
	 * The HairContactsImpulseSolver needs the SegmentKNNSolver for finding the k nearest neighbour 
	 * The HairContactsImpulseSolver needs the SegmentKNNSolver for finding the k nearest neighbour
	 * and add forces between them.
	 */
	class HairContactsImpulseSolver: public Solver {

		friend class HairContactsAndCollisionImpulseSolverVisualizer;

	HairEngine_Public:
		/**
		 * Constructor
		 *
		 * @param creatingDistance The creating distance of contact spring 
		 * @param breakingDistance The breaking distance of the contact spring
		 * @param maxContactPerSegment The limitation of max contact per segment
		 * @param kContactSpring The stiffness of the contact
		 * @param resolution Not use, just a placeholder to make it as the same argument template as the cuda solver
		 * @param wrapSize Not use, placeholder
		 */
		HairContactsImpulseSolver(float creatingDistance, float breakingDistance, int maxContactPerSegment, float kContactSpring, float resolution = 1.0, int wrapSize = 8):
			segmentKnnSolver(nullptr),
			kContactSpring(kContactSpring),
			creatingDistance(creatingDistance),
			breakingDistance(breakingDistance),
			breakingDistanceSquared(breakingDistance * breakingDistance),
			maxContactPerSegment(maxContactPerSegment)
		{}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			// Setup the usedBuffers, assign a individual used buffer for each thread to avoid race condition
			for (int i = 0; i < static_cast<int>(ParallismUtility::getOpenMPMaxHardwareConcurrency()); ++i) {
				usedBuffers.emplace_back(hair.nsegment, -1);
			}

			// Setup the contact springs
			HairEngine_AllocatorAllocate(contactSprings, hair.nsegment * maxContactPerSegment);

			// Setup the ndirected and nundirected
			ncontacts = std::vector<int>(hair.nsegment, 0);

			segmentKnnSolver = new SegmentKNNSolver(creatingDistance);
			segmentKnnSolver->setup(hair, currentTransform);
		}

		void tearDown() override {
			HairEngine_AllocatorDeallocate(contactSprings, hair->nsegment * maxContactPerSegment);

			segmentKnnSolver->tearDown();
			delete segmentKnnSolver;
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {

			std::cout << "[HairContactsImpulseSolver]: Build knn..." << std::endl;

			segmentKnnSolver->solve(hair, info);

			std::cout << "[HairContactsImpulseSolver]: Get impulses..." << std::endl;

			ParallismUtility::parallelFor(0, ParallismUtility::getOpenMPMaxHardwareConcurrency(), [this] (int i){
				std::fill(usedBuffers[i].begin(), usedBuffers[i].end(), -1);
			});

			// Erase all the distance larger the r, don't parallel since we modify nundirected[_.idx2]
			ParallismUtility::parallelForWithThreadIndex(0, hair.nsegment, [this, &hair] (int idx1, int threadId) {
				const auto range = getContactSpringRange(idx1);
				auto seg1 = hair.segments + idx1;
				auto & usedBuffer = usedBuffers[threadId];

				const auto removeEnd = std::remove_if(range.first, range.second, [this, seg1, &hair](const ContactSpringInfo & _) -> bool {
					return (seg1->midpoint() - (hair.segments + _.idx2)->midpoint()).squaredNorm() > breakingDistanceSquared;
				});

				// Update the nContacts now
				ncontacts[idx1] = static_cast<int>(removeEnd - range.first);

				// Compute the force of all undeleted spring and set the usedBuffer
				Eigen::Vector3f force = Eigen::Vector3f::Zero();

				for (auto _ = range.first; _ != removeEnd; ++_) {
					auto seg2 = hair.segments + _->idx2;
					force += MathUtility::massSpringForce(seg1->midpoint(), seg2->midpoint(), kContactSpring, _->l0);

					usedBuffer[_->idx2] = idx1;
				}

				syncLock.lock();
				seg1->p1->impulse += force;
				seg1->p2->impulse += force;
				syncLock.unlock();

				// Add addtional spring
				const int nneeds = std::min(segmentKnnSolver->getNNeighbourForSegment(idx1), maxContactPerSegment - ncontacts[idx1]);
				for (int i = 0; i < nneeds; ++i) {
					const int idx2 = segmentKnnSolver->getNeighbourIndexForSegment(idx1, i);
					const auto seg2 = hair.segments + idx2;

					if (usedBuffer[idx2] != idx1 && seg2->strandIndex() != seg1->strandIndex()) {
						const float l02 = (seg2->midpoint() - seg1->midpoint()).squaredNorm();
						if (l02 < creatingDistance * creatingDistance)
							std::allocator<ContactSpringInfo>().construct(range.first + (ncontacts[idx1]++), idx2, creatingDistance);
					}
				}
			});

			// Summary
			int totalContacts = 0;
			for (int i = 0; i < ncontacts.size(); ++i)
				totalContacts += ncontacts[i];

			std::cout << "[HairContactsImpulseSolver]: Total contacts: " << totalContacts << ", average contacts: " << static_cast<float>(totalContacts) / hair.nsegment << std::endl;
		}

	HairEngine_Protected :

		struct ContactSpringInfo {
			int idx2; ///< The index for another endpoint
			float l0; ///< The rest length when creating 

			ContactSpringInfo(int idx2, float l0): idx2(idx2), l0(l0) {}
		};

		SegmentKNNSolver *segmentKnnSolver;
		float kContactSpring; ///< The stiffness of the contact spring

		ContactSpringInfo *contactSprings; ///< Index array of the contacts spring
		std::vector<std::vector<int>> usedBuffers; ///< Used in iteration to indicate whether the spring has been created
		std::vector<int> ncontacts; ///< How many contact spring is stored in the range (contactSprings[i * maxContactPerSegment], contactSpring[(i+1) * maxContactPerSegment] )

		float creatingDistance;
		float breakingDistance;
		float breakingDistanceSquared;
		int maxContactPerSegment;

		//FIXME
		CompactNSearch::Spinlock syncLock; ///< Use to sync the thread

		std::pair<ContactSpringInfo *, ContactSpringInfo *> getContactSpringRange(int segmentIndex) const {
			std::pair<ContactSpringInfo *, ContactSpringInfo *> ret;
			ret.first = contactSprings + segmentIndex * maxContactPerSegment;;
			ret.second = ret.first + ncontacts[segmentIndex];

			return ret;
		}
	};
};
