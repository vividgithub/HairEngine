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

			if (enableParallism) {
				mapStrand(false, [this, &info, f1, f2, vel, pos, outVel](size_t si) {
					_integrate(pos, vel, outVel, info, f1, f2, static_cast<int>(si));
				});
			}
			else {
				// Integrate all together
				_integrate(pos, vel, outVel, info, f1, f2, -1);
			}

		}

	HairEngine_Protected:
		bool enableParallism;

		void _integrate(Eigen::Vector3f* pos, Eigen::Vector3f* vel, Eigen::Vector3f* outVel, const IntegrationInfo& info, float f1, float f2, int si) {

			size_t n; // Number of particles to solve
			size_t particleStartIndex; // The start index of solving particles
			size_t particleEndIndex; // The end index of the sovling particles
			Spring *springStartPtr, *springEndPtr; // The start and end spring pointer

			if (si >= 0) {
				// True strand
				n = nparticleInStrand[si];
				particleStartIndex = particleStartIndexForStrand[si];
				particleEndIndex = particleStartIndex + n;
				springStartPtr = springs + springStartIndexForStrand[si];
				springEndPtr = springStartPtr + nspringInStrand[si];
			}
			else {
				// Solve all together
				n = nparticle;
				particleStartIndex = 0;
				particleEndIndex = nparticle;
				springStartPtr = springs;
				springEndPtr = springs + nspring;
			}

			Eigen::VectorXf b(n * 3);
			Eigen::SparseMatrix<float> A(3 * n, 3 * n);
			std::vector<Eigen::Triplet<float>> triplets;

			// Helper function
			const auto & getVectorIndex = [particleStartIndex](size_t i) -> int {
				return static_cast<int>(3 * (i - particleStartIndex));
			};
			const auto & isStrandRoot = [this](size_t i) -> bool {
				return isNormalParticle(i) && p(i)->localIndex == 0;
			};

			for (auto i = particleStartIndex; i < particleEndIndex; ++i) {
				const auto i3 = getVectorIndex(i);
				auto par = p(i);

				// If it is not the strand root, assign the triplets and b 
				if (!isStrandRoot(i)) {
					b.segment<3>(i3) = pmass * vel[i] + par->impulse * info.t;

					triplets.emplace_back(i3, i3, f2);
					triplets.emplace_back(i3 + 1, i3 + 1, f2);
					triplets.emplace_back(i3 + 2, i3 + 2, f2);
				}
			}

			for (auto sp = springStartPtr; sp != springEndPtr; ++sp) {

				Eigen::Matrix3f dm;
				const Eigen::Vector3f springImpluse = info.t * MathUtility::massSpringForce(pos[sp->i1], pos[sp->i2], sp->k, sp->l0, nullptr, &dm);

				const auto i1_3 = getVectorIndex(sp->i1);
				const auto i2_3 = getVectorIndex(sp->i2);

				b.segment<3>(i1_3) += springImpluse;
				b.segment<3>(i2_3) -= springImpluse;

				Eigen::Matrix3f vm = (f1 * sp->k) * dm; //Velocity matrix

				// Check whether i1 is the strand root, if so, we don't fill that row
				std::vector<int> rows = { i2_3, i1_3 };
				std::vector<int> cols = { i1_3, i2_3 };
				if (isStrandRoot(sp->i1))
					rows.erase(rows.end() - 1);

				for (auto row : rows)
					for (auto col : cols)
						for (int i = 0; i < 3; ++i)
							for (int j = 0; j < 3; ++j)
								triplets.emplace_back(row + i, col + j, (row == col ? 1.0f : -1.0f) * vm(i, j));
			}

			// Clear the first row and make the velocity match the body transform velocity
			for (size_t i = particleStartIndex; i < particleEndIndex; ++i) if (isStrandRoot(i)) {
				const auto i3 = getVectorIndex(i);

				triplets.emplace_back(i3, i3, 1.0f);
				triplets.emplace_back(i3 + 1, i3 + 1, 1.0f);
				triplets.emplace_back(i3 + 2, i3 + 2, 1.0f);

				auto par = p(i);
				b.segment<3>(i3) = (info.transform * par->restPos - info.previousTransform * par->restPos) / info.t;
			}


			// Conjugate solver
			A.setFromTriplets(triplets.begin(), triplets.end());
			Eigen::ConjugateGradient<Eigen::SparseMatrix<float>> cg;
			cg.compute(A);
			Eigen::VectorXf x = cg.solve(b);

			// Assign back
			for (auto i = particleStartIndex; i < particleEndIndex; ++i) {
				const auto i3 = getVectorIndex(i);
				outVel[i] = x.segment<3>(i3);
			}
		}
	};
}