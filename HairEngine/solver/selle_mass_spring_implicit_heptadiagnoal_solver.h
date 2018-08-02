#pragma once 

#include <algorithm>

#include "selle_mass_spring_solver_base.h"
#include "../util/parallutil.h"

namespace HairEngine {
	/**
	 * The heptadiagnoal band matrix solver for selle mass sprign systems
	 */
	class SelleMassSpringImplcitHeptadiagnoalSolver: public SelleMassSpringSolverBase {

	HairEngine_Public:

		/**
		 * The data buffer that uses for integration, so that we won't allocate the deallocate the buffer
		 * for each integration. We use individual intermediate data buffer for each threads no data conflict 
		 * will happen. We robustly allocate the capcity to the maximum particle numbers in one strand for the 
		 * specific hair geometry.
		 */
		struct IntermediateDataBuffer {
			Eigen::Matrix3f *A[7]; // The input matrix of A.x = b, each row only has 7 non-zero 3x3 block matrix
			Eigen::Matrix3f *L[5]; // Lower decomposition, L[4] is the inverse of L[0], each row only has 4 non-zero 3x3 block matrix
			Eigen::Matrix3f *U[3]; // Upper decompostion, each row only has 4 non-zero 3x3 block matrix, but since the diagnoal is always identity matrix, we don't store it
			Eigen::Vector3f *y, *b;

			size_t capcity; // The number of data that has been allocated

			IntermediateDataBuffer(size_t capcity): capcity(capcity) {
				for (size_t i = 0; i < 7; ++i)
					A[i] = new Eigen::Matrix3f[capcity];
				for (size_t i = 0; i < 5; ++i)
					L[i] = new Eigen::Matrix3f[capcity];
				for (size_t i = 0; i < 3; ++i)
					U[i] = new Eigen::Matrix3f[capcity];

				y = new Eigen::Vector3f[capcity];
				b = new Eigen::Vector3f[capcity];
			}

			~IntermediateDataBuffer() {
				for (size_t i = 0; i < 7; ++i)
					delete[] A[i];
				for (size_t i = 0; i < 5; ++i)
					delete[] L[i];
				for (size_t i = 0; i < 3; ++i)
					delete[] U[i];

				delete[] y;
				delete[] b;
			}
		};

		using SelleMassSpringSolverBase::SelleMassSpringSolverBase;

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			SelleMassSpringSolverBase::setup(hair, currentTransform);

			// Find the maxiumu particle size
			maxParticleSize = 0;
			for (size_t i = 0; i < nstrand; ++i)
				maxParticleSize = std::max(maxParticleSize, nparticleInStrand[i]);

			// Allocate the data buffer for each thread
			nDataBuffer = ParallismUtility::getOpenMPMaxHardwareConcurrency();
			HairEngine_AllocatorAllocate(dataBuffers, nDataBuffer);
			for (size_t i = 0; i < nDataBuffer; ++i)
				std::allocator<IntermediateDataBuffer>().construct(dataBuffers + i, maxParticleSize);
		}

		void tearDown() override {
			
			HairEngine_AllocatorDeallocate(dataBuffers, nDataBuffer);

			SelleMassSpringSolverBase::tearDown();
		}

		void integrate(Eigen::Vector3f* pos, Eigen::Vector3f* vel, Eigen::Vector3f* outVel, const IntegrationInfo& info) override {\

			// Pre computed data
			const float f1 = info.t * info.t;
			const Eigen::Matrix3f f2 = (pmass + damping * info.t) * Eigen::Matrix3f::Identity();

			// Parallel solve for each strand
			ParallismUtility::parallelForWithThreadIndex(0, static_cast<int>(nstrand), [this, pos, vel, outVel, &info, &f1, &f2] (int si, int threadID) {
				auto & _ = dataBuffers[threadID];

				const size_t particleStartIndex = particleStartIndexForStrand[si];
				const size_t n = nparticleInStrand[si];
				const size_t particleEndIndex = particleStartIndex + n;
				const Spring *springStartPtr = springs + springStartIndexForStrand[si];
				const Spring *springEndPtr = springStartPtr + nspringInStrand[si];

				const Eigen::Matrix3f & zero = Eigen::Matrix3f::Zero();
				const Eigen::Matrix3f & identity = Eigen::Matrix3f::Identity();

				// Initialize the b and A
				for (size_t i = particleStartIndex + 1; i != particleEndIndex; ++i) {
					size_t vi = i - particleStartIndex; // Vector(matrix) index

					_.A[3][vi] = f2;
					_.A[0][vi] = _.A[1][vi] = _.A[2][vi] = _.A[4][vi] = _.A[5][vi] = _.A[6][vi] = zero;
					_.b[vi] = pmass * vel[i] + p(i)->impulse * info.t;
				}

				// Spring forces and direction matrix 
				for (auto sp = springStartPtr; sp != springEndPtr; ++sp) {
					Eigen::Matrix3f dm;
					const Eigen::Vector3f springImpluse = info.t * MathUtility::massSpringForce(pos[sp->i1], pos[sp->i2], sp->k, sp->l0, nullptr, &dm);
					const Eigen::Matrix3f vm = (f1 * sp->k) * dm; //Velocity matrix

					const int diff = static_cast<int>(sp->i2) - static_cast<int>(sp->i1);

					const size_t vi1 = sp->i1 - particleStartIndex;
					const size_t vi2 = sp->i2 - particleStartIndex;

					_.b[vi1] += springImpluse;
					_.b[vi2] -= springImpluse;

					_.A[3][vi1] += vm;
					_.A[3][vi2] += vm;
					_.A[3 + diff][vi1] = -vm;
					_.A[3 - diff][vi2] = -vm;
				}

				// Initialize the b and A of the strand root
				auto rootPar = p(particleStartIndex);
				_.b[0] = (info.transform * rootPar->restPos - info.previousTransform * rootPar->restPos) / info.t;
				_.A[0][0] = _.A[1][0] = _.A[2][0] = _.A[4][0] = _.A[5][0] = _.A[6][0] = zero;
				_.A[3][0] = identity;

				// Heptadignoal solver
				for (size_t i = 0; i < n; ++i) {
					//compute L3
					_.L[3][i] = i >= 3 ? _.A[0][i] : zero;

					//compute L2
					_.L[2][i] = i >= 2 ? _.A[1][i] : zero;
					if (i >= 3)
						_.L[2][i] -= _.L[3][i] * _.U[0][i - 3];

					//compute L1
					_.L[1][i] = i >= 1 ? _.A[2][i] : zero;
					if (i >= 2)
						_.L[1][i] -= _.L[2][i] * _.U[0][i - 2];
					if (i >= 3)
						_.L[1][i] -= _.L[3][i] * _.U[1][i - 3];

					//compute L0
					_.L[0][i] = _.A[3][i];
					if (i >= 1)
						_.L[0][i] -= _.L[1][i] * _.U[0][i - 1];
					if (i >= 2)
						_.L[0][i] -= _.L[2][i] * _.U[1][i - 2];
					if (i >= 3)
						_.L[0][i] -= _.L[3][i] * _.U[2][i - 3];

					//compute L0i
					_.L[4][i] = _.L[0][i].inverse();

					//compute U2
					_.U[2][i] = i + 3 < n ? _.L[4][i] * _.A[6][i] : zero;

					//compute U1
					_.U[1][i] = i + 2 < n ? _.A[5][i] : zero;
					if (i >= 1)
						_.U[1][i] -= _.L[1][i] * _.U[2][i - 1];
					_.U[1][i] = _.L[4][i] * _.U[1][i];

					//compute U0
					_.U[0][i] = i + 1 < n ? _.A[4][i] : zero;
					if (i >= 1)
						_.U[0][i] -= _.L[1][i] * _.U[1][i - 1];
					if (i >= 2)
						_.U[0][i] -= _.L[2][i] * _.U[2][i - 2];
					_.U[0][i] = _.L[4][i] * _.U[0][i];

					//compute y
					_.y[i] = _.b[i];
					if (i >= 1)
						_.y[i] -= _.L[1][i] * _.y[i - 1];
					if (i >= 2)
						_.y[i] -= _.L[2][i] * _.y[i - 2];
					if (i >= 3)
						_.y[i] -= _.L[3][i] * _.y[i - 3];
					_.y[i] = _.L[4][i] * _.y[i];
				}

				// Compute the final velocity
				auto x = outVel + particleStartIndex;
				for (int i = n - 1; i >= 0; --i) {
					x[i] = _.y[i];
					if (i + 1 < n)
						x[i] -= _.U[0][i] * x[i + 1];
					if (i + 2 < n)
						x[i] -= _.U[1][i] * x[i + 2];
					if (i + 3 < n)
						x[i] -= _.U[2][i] * x[i + 3];
				}

				// Debug
				//for (size_t i = 0; i < n; ++i) {
				//	std::cout << "A=" << std::endl;
				//	for (size_t k = 0; k < 7; ++k)
				//		std::cout << "A[" << k << "][" << i << "]=\n" << _.A[k][i] << std::endl;
				//}

				//for (size_t i = 0; i < n; ++i) {
				//	std::cout << "b=" << std::endl;
				//	std::cout << "b[" << i << "]=" << _.b[i] << std::endl;
				//}

				//for (size_t i = 0; i < n; ++i) {
				//	std::cout << "L=" << std::endl;
				//	for (size_t k = 0; k < 5; ++k)
				//		std::cout << "L[" << k << "][" << i << "]=\n" << _.L[k][i] << std::endl;
				//}

				//for (size_t i = 0; i < n; ++i) {
				//	std::cout << "U=" << std::endl;
				//	for (size_t k = 0; k < 3; ++k)
				//		std::cout << "U[" << k << "][" << i << "]=\n" << _.U[k][i] << std::endl;
				//}

				//for (size_t i = 0; i < n; ++i) {
				//	std::cout << "y=" << std::endl;
				//	std::cout << "y[" << i << "]=" << _.y[i] << std::endl;
				//}

				//for (size_t i = 0; i < n; ++i) {
				//	std::cout << "x=" << std::endl;
				//	std::cout << "x[" << i << "]=" << x[i] << std::endl;
				//}

				//std::cout << "-----------------------" << std::endl;
			});
		}

	HairEngine_Protected:
		IntermediateDataBuffer *dataBuffers = nullptr;
		size_t nDataBuffer = 0;
		size_t maxParticleSize = 0;
	};
}