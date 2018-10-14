#pragma once

#include <array>

#include "hair_contacts_impulse_solver.h"
#include "collision_impulse_solver.h"
#include "visualizer.h"

namespace HairEngine {
	/**
	* The visualizer solver for hair contacts impulse solver. It tries to write every hair contacts
	* as an line in the vply file.
	*/
	class HairContactsAndCollisionImpulseSolverVisualizer : public Visualizer {

	HairEngine_Public:
		HairContactsAndCollisionImpulseSolverVisualizer(const std::string & directory, const std::string & filenameTemplate,
			float timestep, HairContactsImpulseSolver *hairContactsImpulseSolver = nullptr, CollisionImpulseSolver *collisionImpulseSolver = nullptr):
			Visualizer(directory, filenameTemplate, timestep), 
			hairContactsImpulseSolver(hairContactsImpulseSolver), 
			collisionImpulseSolver(collisionImpulseSolver) {}

		void visualize(std::ostream& os, Hair& hair, const IntegrationInfo& info) override {
			if (hairContactsImpulseSolver)
				visualizeHairContacts(os, hair, info);
			if (collisionImpulseSolver)
				visualizeCollisions(os, hair, info);
		}

	HairEngine_Protected:
		HairContactsImpulseSolver * hairContactsImpulseSolver; ///< The referenced hair contacts impulse solver
		CollisionImpulseSolver *collisionImpulseSolver; ///< The referenced collision impulse solver

		void visualizeHairContacts(std::ostream& os, Hair& hair, const IntegrationInfo& info) const {
			int totalContacts = 0;

			for (int idx1 = 0; idx1 < hair.nsegment; ++idx1) {
				totalContacts += static_cast<int>(hairContactsImpulseSolver->ncontacts[idx1]);
				auto range = hairContactsImpulseSolver->getContactSpringRange(idx1);

				for (auto _ = range.first; _ != range.second; ++_) {

					std::array<Eigen::Vector3f, 2> midpoints = {
						(hair.segments + idx1)->midpoint(),
						(hair.segments + _->idx2)->midpoint()
					};

					VPly::writeLine(
						os,
						EigenUtility::toVPlyVector3f(midpoints[0]),
						EigenUtility::toVPlyVector3f(midpoints[1]),
						VPly::VPlyFloatAttr("l0", _->l0),
						VPly::VPlyFloatAttr("l", (midpoints[1] - midpoints[0]).norm()),
						VPly::VPlyIntAttr("fromid", idx1),
						VPly::VPlyIntAttr("toid", _->idx2),
						VPly::VPlyIntAttr("type", 10)
					);
				}
			}

			std::cout << "HairContactsAndCollisionImpulseVisualizer: Total contacts=" << totalContacts
				<< ", Average contacts per particle=" << static_cast<float>(totalContacts) / hair.nparticle << std::endl;
		}

		void visualizeCollisions(std::ostream& os, Hair& hair, const IntegrationInfo& info) const {
//			int totalCollisions = 0;
//
//			for (int idx1 = 0; idx1 < hair.nsegment; ++idx1) {
//				totalCollisions += static_cast<int>(collisionImpulseSolver->ncollision[idx1]);
//				auto range = collisionImpulseSolver->getCollisionRange(idx1);
//
//				for (auto _ = range.first; _ != range.second; ++_) {
//
//					std::array<Eigen::Vector3f, 2> midpoints = {
//						(hair.segments + idx1)->lerpPos(_->t1),
//						(hair.segments + _->idx2)->lerpPos(_->t2)
//					};
//
//					VPly::writeLine(
//						os,
//						EigenUtility::toVPlyVector3f(midpoints[0]),
//						EigenUtility::toVPlyVector3f(midpoints[1]),
//						VPly::VPlyFloatAttr("l0", _->l0),
//						VPly::VPlyFloatAttr("l", (midpoints[1] - midpoints[0]).norm()),
//						VPly::VPlyIntAttr("fromid", idx1),
//						VPly::VPlyIntAttr("toid", _->idx2),
//						VPly::VPlyIntAttr("type", 11)
//					);
//				}
//			}
//
//			std::cout << "HairContactsImpulseAndCollisionVisualizer: Total collisions=" << totalCollisions
//				<< ", Average collisions per particle=" << static_cast<float>(totalCollisions) / hair.nparticle << std::endl;
		}
	};
}