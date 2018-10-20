#pragma once

#include "VPly/vply.h"
#include "selle_mass_spring_solver_base.h"

namespace HairEngine {
	class SelleMassSpringVisualizer: public Visualizer {
	HairEngine_Public:

		/**
		 * Constructor
		 * 
		 * @param directory Describe in Visualizer
		 * @param filenameTemplate Describe in Visualizer
		 * @param selleMassSpringSolver The observing solver
		 */
		SelleMassSpringVisualizer(const std::string & directory, const std::string & filenameTemplate, float timestep, SelleMassSpringSolverBase * selleMassSpringSolver):
			Visualizer(directory, filenameTemplate, timestep), selleMassSpringSolver(selleMassSpringSolver) {}

		void visualize(std::ostream& os, Hair& hair, const IntegrationInfo& info) override {

			if (!selleMassSpringSolver)
				return;

			// Show the springs
			const auto _ = selleMassSpringSolver;
			for (auto sp = _->springs; sp != _->springs + _->nspring; ++sp) {
				Eigen::Vector3f pos1 = _->p(sp->i1)->pos;
				Eigen::Vector3f pos2 = _->p(sp->i2)->pos;
				float l = (pos2 - pos1).norm();

				float rigidness = (selleMassSpringSolver->particleProps[sp->i1].rigidness +
						selleMassSpringSolver->particleProps[sp->i2].rigidness) / 2.0f;

				VPly::writeLine(
					os, EigenUtility::toVPlyVector3f(pos1), EigenUtility::toVPlyVector3f(pos2),
					VPly::VPlyFloatAttr("k", sp->k),
					VPly::VPlyFloatAttr("l0", sp->l0),
					VPly::VPlyFloatAttr("l", l),
					VPly::VPlyFloatAttr("rg", rigidness),
					VPly::VPlyIntAttr("type", static_cast<int32_t>(sp->typeID))
				);
			}
		}

	HairEngine_Protected:
		SelleMassSpringSolverBase *selleMassSpringSolver; ///< The observing solver
	};
}