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

			const float f1 = info.t * info.t;
			const float f2 = pmass + damping * info.t;

			mapStrand(true, [this, &info, f1, f2, vel, pos, outVel](size_t si)
			{
				size_t n = nparticleInStrand[si];

				Eigen::VectorXf b(n * 3);
				Eigen::SparseMatrix<float> A(3 * n, 3 * n);
				std::vector<Eigen::Triplet<float>> triplets;

				// Iteration ignore the strand root
				for (auto i = particleStartIndexForStrand[si] + 1; i < particleStartIndexForStrand[si] + n; ++i) {
					const auto i3 = static_cast<int>((i - particleStartIndexForStrand[si]) * 3);
					auto par = p(i);

					//A.block<3, 3>(i3, i3) = (pmass + damping * info.t) * Eigen::Matrix3f::Identity();
					b.segment<3>(i3) = pmass * vel[i] + par->impulse * info.t;

					triplets.emplace_back(i3, i3, f2);
					triplets.emplace_back(i3 + 1, i3 + 1, f2);
					triplets.emplace_back(i3 + 2, i3 + 2, f2);
				}


				for (auto sp = springs + springStartIndexForStrand[si]; sp != springs + springStartIndexForStrand[si] + nspringInStrand[si]; ++sp) {

					Eigen::Matrix3f dm;
					const Eigen::Vector3f springImpluse = info.t * MathUtility::massSpringForce(pos[sp->i1], pos[sp->i2], sp->k, sp->l0, nullptr, &dm);

					const auto i1_3 = static_cast<int>(3 * (sp->i1 - particleStartIndexForStrand[si]));
					const auto i2_3 = static_cast<int>(3 * (sp->i2 - particleStartIndexForStrand[si]));

					b.segment<3>(i1_3) += springImpluse;
					b.segment<3>(i2_3) -= springImpluse;

					Eigen::Matrix3f vm = (f1 * sp->k) * dm; //Velocity matrix

					// Check whether i1 is the strand root, if so, we don't fill that row
					std::vector<int> rows = { i2_3, i1_3 };
					std::vector<int> cols = { i1_3, i2_3 };
					if (isNormalParticle(sp->i1) && p(sp->i1)->localIndex == 0)
						rows.erase(rows.end() - 1);
					for (auto row : rows)
						for (auto col : cols)
							for (int i = 0; i < 3; ++i)
								for (int j = 0; j < 3; ++j)
									triplets.emplace_back(row + i, col + j, (row == col ? 1.0f : -1.0f) * vm(i, j));
					
					//A.block<3, 3>(i1_3, i1_3) += vm;
					//A.block<3, 3>(i1_3, i2_3) -= vm;
					//A.block<3, 3>(i2_3, i2_3) += vm;
					//A.block<3, 3>(i2_3, i1_3) -= vm;
				}

				// Clear the first row and make the velocity match the body transform velocity
				triplets.emplace_back(0, 0, 1.0f);
				triplets.emplace_back(1, 1, 1.0f);
				triplets.emplace_back(2, 2, 1.0f);
				auto rootPar = p(particleStartIndexForStrand[si]);
				b.segment<3>(0) = (info.transform * rootPar->restPos - info.previousTransform * rootPar->restPos) / info.t;

				//Conjugate solver
				A.setFromTriplets(triplets.begin(), triplets.end());
				Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> cg;
				cg.compute(A);
				Eigen::VectorXf x = cg.solve(b);

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