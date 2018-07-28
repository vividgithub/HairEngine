#pragma once

#include <vector>

#include "../util/mathutil.h"
#include "selle_mass_spring_solver_base.h"

namespace HairEngine {
	class SelleMassSpringImplicitSolver: public SelleMassSpringSolverBase {
	HairEngine_Public:

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			SelleMassSpringSolverBase::setup(hair, currentTransform);
		}

		void tearDown() override {
			SelleMassSpringSolverBase::tearDown();
		}

		void integrate(Eigen::Vector3f* pos, Eigen::Vector3f* vel, Eigen::Vector3f* outVel, const IntegrationInfo& info) override {
			std::vector<Eigen::Triplet<float>> triplets;

			// Iterate through springs
			for (auto sp = springs; sp != springs + nspring; ++sp) {
				auto p1 = p(sp->i1);
				auto p2 = p(sp->i2);

				Eigen::Matrix3f dm; // Direction matrix
				Eigen::Vector3f springForce = MathUtility::massSpringForce(p1->pos, p2->pos, sp->k, sp->l0, nullptr, &dm);


			}
		}
	};
}