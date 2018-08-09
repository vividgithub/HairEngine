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

				VPly::writeLine(
					os, EigenUtility::toVPlyVector3f(pos1), EigenUtility::toVPlyVector3f(pos2),
					VPly::VPlyFloatAttr("k", sp->k),
					VPly::VPlyFloatAttr("l0", sp->l0),
					VPly::VPlyFloatAttr("l", l),
					VPly::VPlyIntAttr("type", static_cast<int32_t>(sp->typeID))
				);
			}

			Eigen::Vector3f normals[7], vs[7];
			for (auto sp = _->altitudeSprings; sp != _->altitudeSprings + _->naltitude; ++sp) {
				int selectedIndex = _->getAltitudeSpringSelectedSpringIndex(sp, normals, vs);

				auto p1 = _->p(sp->i1), p2 = _->p(sp->i2), p3 = _->p(sp->i3), p4 = _->p(sp->i4);

				normals[selectedIndex].normalize();
				Eigen::Vector3f d = MathUtility::project(vs[selectedIndex], normals[selectedIndex]);

				//Make the direction forward to p1 -> another point
				if (d.dot(vs[selectedIndex]) < 0)
					d = -d;

				float l = d.norm();

				switch (selectedIndex) {
				case 0: // {p1,p2} -> {p3, p4}
					VPly::writeLine(
						os,
						EigenUtility::toVPlyVector3f(MathUtility::midPoint(p1->pos, p2->pos)),
						EigenUtility::toVPlyVector3f(MathUtility::midPoint(p3->pos, p4->pos)),
						VPly::VPlyFloatAttr("k", sp->k),
						VPly::VPlyFloatAttr("l", l),
						VPly::VPlyFloatAttr("l0", sp->l0s[selectedIndex]),
						VPly::VPlyIntAttr("type", static_cast<int32_t>(3))
					);
					break;
				case 1: // {p1, p3} -> {p2, p4}
					VPly::writeLine(
						os,
						EigenUtility::toVPlyVector3f(MathUtility::midPoint(p1->pos, p3->pos)),
						EigenUtility::toVPlyVector3f(MathUtility::midPoint(p2->pos, p4->pos)),
						VPly::VPlyFloatAttr("k", sp->k),
						VPly::VPlyFloatAttr("l", l),
						VPly::VPlyFloatAttr("l0", sp->l0s[selectedIndex]),
						VPly::VPlyIntAttr("type", static_cast<int32_t>(3))
					);
					break;
				case 2: // {p1, p4} -> {p2, p3}
					VPly::writeLine(
						os,
						EigenUtility::toVPlyVector3f(MathUtility::midPoint(p1->pos, p4->pos)),
						EigenUtility::toVPlyVector3f(MathUtility::midPoint(p2->pos, p3->pos)),
						VPly::VPlyFloatAttr("k", sp->k),
						VPly::VPlyFloatAttr("l", l),
						VPly::VPlyFloatAttr("l0", sp->l0s[selectedIndex]),
						VPly::VPlyIntAttr("type", static_cast<int32_t>(3))
					);
					break;
				case 3: // {p1} -> {p2, p3, p4}
					VPly::writeLine(
						os,
						EigenUtility::toVPlyVector3f(p1->pos),
						EigenUtility::toVPlyVector3f(MathUtility::triangleCenter(p2->pos, p3->pos, p4->pos)),
						VPly::VPlyFloatAttr("k", sp->k),
						VPly::VPlyFloatAttr("l", l),
						VPly::VPlyFloatAttr("l0", sp->l0s[selectedIndex]),
						VPly::VPlyIntAttr("type", static_cast<int32_t>(3))
					);
					break;
				case 4: // {p2} -> {p1, p3, p4}
					VPly::writeLine(
						os,
						EigenUtility::toVPlyVector3f(p2->pos),
						EigenUtility::toVPlyVector3f(MathUtility::triangleCenter(p1->pos, p3->pos, p4->pos)),
						VPly::VPlyFloatAttr("k", sp->k),
						VPly::VPlyFloatAttr("l", l),
						VPly::VPlyFloatAttr("l0", sp->l0s[selectedIndex]),
						VPly::VPlyIntAttr("type", static_cast<int32_t>(3))
					);
					break;
				case 5: // {p3} -> {p1, p2, p4}
					VPly::writeLine(
						os,
						EigenUtility::toVPlyVector3f(p3->pos),
						EigenUtility::toVPlyVector3f(MathUtility::triangleCenter(p1->pos, p2->pos, p4->pos)),
						VPly::VPlyFloatAttr("k", sp->k),
						VPly::VPlyFloatAttr("l", l),
						VPly::VPlyFloatAttr("l0", sp->l0s[selectedIndex]),
						VPly::VPlyIntAttr("type", static_cast<int32_t>(3))
					);
					break;
				default: // {p4} -> {p1, p2, p3}
					VPly::writeLine(
						os,
						EigenUtility::toVPlyVector3f(p4->pos),
						EigenUtility::toVPlyVector3f(MathUtility::triangleCenter(p1->pos, p2->pos, p3->pos)),
						VPly::VPlyFloatAttr("k", sp->k),
						VPly::VPlyFloatAttr("l", l),
						VPly::VPlyFloatAttr("l0", sp->l0s[selectedIndex]),
						VPly::VPlyIntAttr("type", static_cast<int32_t>(3))
					);
					break;
				}
			}
		}

	HairEngine_Protected:
		SelleMassSpringSolverBase *selleMassSpringSolver; ///< The observing solver
	};
}