#pragma once

#include <vector>
#include <iostream>
#include "Eigen/Sparse"

#include "../util/mathutil.h"
#include "selle_mass_spring_solver_base.h"

namespace HairEngine {
	class SelleMassSpringImplicitSolver: public SelleMassSpringSolverBase {

	HairEngine_Public:
		/**
		 * Constructor
		 * 
		 * @param conf The SelleMassSpringSolverBase configuration
		 * @param enableParallism True to enable parallel solving for each strand, false to solve in a big linear equations
		 */
		SelleMassSpringImplicitSolver(const Configuration & conf, bool enableParallism = true):
			SelleMassSpringSolverBase(conf), enableParallism(enableParallism) {}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			SelleMassSpringSolverBase::setup(hair, currentTransform);
		}

		void tearDown() override {
			SelleMassSpringSolverBase::tearDown();
		}

		void integrate(Eigen::Vector3f* pos, Eigen::Vector3f* vel, Eigen::Vector3f* outVel, const IntegrationInfo& info) override {

			mapStrand(true, [this, &info, vel, pos, outVel](size_t si)
			{
				size_t n = nparticleInStrand[si];

				Eigen::VectorXf b(n * 3), x(n * 3);
				Eigen::MatrixXf A(n * 3, n * 3);
				A.setZero();

				for (auto i = particleStartIndexForStrand[si]; i < particleStartIndexForStrand[si] + n; ++i) {
					const size_t i3 = (i - particleStartIndexForStrand[si]) * 3;
					auto par = p(i);

					A.block<3, 3>(i3, i3) = (pmass + damping * info.t) * Eigen::Matrix3f::Identity();
					b.segment<3>(i3) = pmass * vel[i] + par->impulse * info.t;
				}


				for (auto sp = springs + springStartIndexForStrand[si]; sp != springs + springStartIndexForStrand[si] + nspringInStrand[si]; ++sp) {

					Eigen::Matrix3f dm;
					const Eigen::Vector3f springImpluse = info.t * MathUtility::massSpringForce(pos[sp->i1], pos[sp->i2], sp->k, sp->l0, nullptr, &dm);

					size_t i1_3 = 3 * (sp->i1 - particleStartIndexForStrand[si]);
					size_t i2_3 = 3 * (sp->i2 - particleStartIndexForStrand[si]);

					b.segment<3>(i1_3) += springImpluse;
					b.segment<3>(i2_3) -= springImpluse;

					Eigen::Matrix3f vm = info.t * (sp->k * info.t) * dm; //Velocity matrix

					A.block<3, 3>(i1_3, i1_3) += vm;
					A.block<3, 3>(i1_3, i2_3) -= vm;
					A.block<3, 3>(i2_3, i2_3) += vm;
					A.block<3, 3>(i2_3, i1_3) -= vm;
				}

				//Clear the first row and make the velocity match the body transform velocity
				for (size_t r = 0; r < 3; ++r)
					for (size_t c = 0; c < A.cols(); ++c)
						A(r, c) = 0.0f;
				A.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();

				auto rootPar = p(particleStartIndexForStrand[si]);
				b.segment<3>(0) = (info.transform * rootPar->restPos
					- info.previousTransform * rootPar->restPos) / info.t;

				//Conjugate solver
				x = A.inverse() * b;

				//Assign back
				for (auto i = particleStartIndexForStrand[si]; i < particleStartIndexForStrand[si] + n; ++i) {
					const size_t i3 = (i - particleStartIndexForStrand[si]) * 3;
					outVel[i] = x.segment<3>(i3);
				}
			});
		}

	HairEngine_Protected:
		bool enableParallism;
	};
}