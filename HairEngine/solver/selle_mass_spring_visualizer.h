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
			// Show the springs
			const auto _ = selleMassSpringSolver;
			for (auto sp = _->springs; sp != _->springs + _->nspring; ++sp) {
				Eigen::Vector3f pos1 = _->p(sp->i1)->pos;
				Eigen::Vector3f pos2 = _->p(sp->i2)->pos;
				float l = (pos2 - pos1).norm();

				Eigen::Vector3f springForce = MathUtility::massSpringForce(pos1, pos2, sp->k, sp->l0);

				VPly::writeLine(
					os, EigenUtility::toVPlyVector3f(pos1), EigenUtility::toVPlyVector3f(pos2),
					VPly::VPlyFloatAttr("k", sp->k),
					VPly::VPlyFloatAttr("l0", sp->l0),
					VPly::VPlyFloatAttr("l", l),
					VPly::VPlyVector3fAttr("fd", EigenUtility::toVPlyVector3f(springForce)),
					VPly::VPlyIntAttr("type", static_cast<int32_t>(sp->typeID))
				);
			}

			Eigen::Vector3f normals[7], vs[7];
			for (auto sp = _->altitudeSprings; sp != _->altitudeSprings + _->naltitude; ++sp) {

				const auto spInfo = _->getAltitudeSpringInfo(sp);

				std::array<Eigen::Vector3f, 2> attachedPoints = { Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero() };

				for (int i = 0; i < 4; ++i) {
					auto & attachedPoint = spInfo.signs[i] > 0.0f ? attachedPoints[0] : attachedPoints[1];
					attachedPoint += spInfo.intp[i] * spInfo.p[i]->pos;
				}
				attachedPoints[1] = -attachedPoints[1]; // The sign in the computation of attachPoints[1] is inverse

				HairEngine_DebugAssert(std::abs((attachedPoints[1] - attachedPoints[0]).norm() - spInfo.l) < 4e-4f);

				Eigen::Vector3f springForce = sp->k * (spInfo.l - spInfo.l0) * spInfo.d;

				//Make the direction forward to p1 -> another point
				VPly::writeLine(os,
					EigenUtility::toVPlyVector3f(attachedPoints[0]),
					EigenUtility::toVPlyVector3f(attachedPoints[1]),
					VPly::VPlyFloatAttr("k", sp->k),
					VPly::VPlyFloatAttr("l0", spInfo.l0),
					VPly::VPlyFloatAttr("l", spInfo.l),
					VPly::VPlyVector3fAttr("fd", EigenUtility::toVPlyVector3f(springForce)),
					VPly::VPlyIntAttr("type", static_cast<int32_t>(spInfo.selectedIndex + 3))
				);
			}
		}

	HairEngine_Protected:
		SelleMassSpringSolverBase *selleMassSpringSolver; ///< The observing solver
	};
}