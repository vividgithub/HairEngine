#pragma once 

#include <algorithm>
#include <array>

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
			Eigen::Matrix4f *A[7]; // The input matrix of A.x = b, each row only has 7 non-zero 3x3 block matrix
			Eigen::Matrix4f *L[5]; // Lower decomposition, L[4] is the inverse of L[0], each row only has 4 non-zero 3x3 block matrix
			Eigen::Matrix4f *U[3]; // Upper decompostion, each row only has 4 non-zero 3x3 block matrix, but since the diagnoal is always identity matrix, we don't store it
			Eigen::Vector4f *y, *b, *x;

			int capcity; // The number of data that has been allocated

			IntermediateDataBuffer(int capcity): capcity(capcity) {
				for (int i = 0; i < 7; ++i)
					A[i] = new Eigen::Matrix4f[capcity];
				for (int i = 0; i < 5; ++i)
					L[i] = new Eigen::Matrix4f[capcity];
				for (int i = 0; i < 3; ++i)
					U[i] = new Eigen::Matrix4f[capcity];

				y = new Eigen::Vector4f[capcity];
				b = new Eigen::Vector4f[capcity];
				x = new Eigen::Vector4f[capcity];
			}

			~IntermediateDataBuffer() {
				for (int i = 0; i < 7; ++i)
					delete[] A[i];
				for (int i = 0; i < 5; ++i)
					delete[] L[i];
				for (int i = 0; i < 3; ++i)
					delete[] U[i];

				delete[] y;
				delete[] b;
				delete[] x;
			}
		};

		using SelleMassSpringSolverBase::SelleMassSpringSolverBase;

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			SelleMassSpringSolverBase::setup(hair, currentTransform);

			// Find the maxiumu particle size
			maxParticleSize = 0;
			for (int i = 0; i < nstrand; ++i)
				maxParticleSize = std::max(maxParticleSize, nparticleInStrand[i]);

			// Allocate the data buffer for each thread
			nDataBuffer = ParallismUtility::getOpenMPMaxHardwareConcurrency();
			HairEngine_AllocatorAllocate(dataBuffers, nDataBuffer);
			for (int i = 0; i < nDataBuffer; ++i)
				std::allocator<IntermediateDataBuffer>().construct(dataBuffers + i, maxParticleSize);
		}

		void tearDown() override {
			
			HairEngine_AllocatorDeallocate(dataBuffers, nDataBuffer);

			SelleMassSpringSolverBase::tearDown();
		}

		void integrate(Eigen::Vector3f* pos, Eigen::Vector3f* vel, Eigen::Vector3f* outVel, const IntegrationInfo& info) override {

			// Pre computed data
			const float f1 = info.t * info.t;
			const Eigen::Matrix4f f2 = (pmass + damping * info.t) * Eigen::Matrix4f::Identity();

			// Parallel solve for each strand
			ParallismUtility::parallelForWithThreadIndex(0, static_cast<int>(nstrand), [this, pos, vel, outVel, &info, &f1, &f2] (int si, int threadID) {
				auto & _ = dataBuffers[threadID];

				const int particleStartIndex = particleStartIndexForStrand[si];
				const int n = nparticleInStrand[si];
				const int particleEndIndex = particleStartIndex + n;

				const Spring *springStartPtr = springs + springStartIndexForStrand[si];
				const Spring *springEndPtr = springStartPtr + nspringInStrand[si];

				AltitudeSpring *altitudeStartPtr = altitudeSprings + altitudeStartIndexForStrand[si];
				AltitudeSpring *altitudeEndPtr = altitudeStartPtr + naltitudeInStrand[si];

				const Eigen::Matrix4f & zero = Eigen::Matrix4f::Zero();
				const Eigen::Matrix4f & identity = Eigen::Matrix4f::Identity();

				// Initialize the b and A
				for (int i = particleStartIndex + 1; i != particleEndIndex; ++i) {
					int vi = i - particleStartIndex; // Vector(matrix) index

					_.A[3][vi] = f2;
					_.A[0][vi] = _.A[1][vi] = _.A[2][vi] = _.A[4][vi] = _.A[5][vi] = _.A[6][vi] = zero;

					_.b[vi].segment<3>(0) = pmass * vel[i] + p(i)->impulse * info.t;
					_.b[vi](3) = 0.0f;  // Always set the last vector to 0
				}

				// Spring forces and direction matrix 
				for (auto sp = springStartPtr; sp != springEndPtr; ++sp) {
					Eigen::Matrix4f dm;
					const Eigen::Vector4f springImpulse = info.t * MathUtility::massSpringForce4f(pos[sp->i1], pos[sp->i2], sp->k, sp->l0, nullptr, &dm);
					dm *= f1 * sp->k;

					const int diff = static_cast<int>(sp->i2) - static_cast<int>(sp->i1);

					const int vi1 = sp->i1 - particleStartIndex;
					const int vi2 = sp->i2 - particleStartIndex;

					_.b[vi1] += springImpulse;
					_.b[vi2] -= springImpulse;

					_.A[3][vi1] += dm;
					_.A[3][vi2] += dm;
					_.A[3 + diff][vi1] = -dm;
					_.A[3 - diff][vi2] = -dm;
				}

				Eigen::Vector3f vs[7], normals[7];
				// Altitude spring forces
				for (auto sp = altitudeStartPtr; sp != altitudeEndPtr; ++sp) {

					const auto spInfo = getAltitudeSpringInfo(sp);
					Eigen::Vector4f springImpulse = Eigen::Vector4f::Zero();
					springImpulse.segment<3>(0) = (info.t * sp->k * (spInfo.l - spInfo.l0)) * spInfo.d;

					Eigen::Matrix4f dm = Eigen::Matrix4f::Zero();
					dm.block<3, 3>(0, 0) = (f1 * sp->k) * spInfo.d * spInfo.d.transpose();

					std::array<int, 4> vis = {
						sp->i1 - particleStartIndex,
						sp->i2 - particleStartIndex,
						sp->i3 - particleStartIndex,
						sp->i4 - particleStartIndex,
					};

					for (int i = 0; i < 4; ++i) {
						const int vi = vis[i];
						// Add the spring impulse
						_.b[vi] += spInfo.signs[i] * springImpulse;

						// Add the direction matrix
						// for (int j = 0; j < 4; ++i) {
						// 	int diff = vis[j] - vi;
						// 	_.A[3 + diff][vi] += spInfo.signs[i] * spInfo.intp[j] * dm;
						// }
					}
				}

				// Initialize the b and A of the strand root
				auto rootPar = p(particleStartIndex);

				Eigen::Vector4f rootParRestPos;
				rootParRestPos.segment<3>(0) = rootPar->restPos;
				rootParRestPos(3) = 0.0f;

				Eigen::Matrix4f deltaTransform = info.transform.matrix() - info.previousTransform.matrix();

				_.b[0] = (deltaTransform * rootParRestPos) / info.t;
				_.A[0][0] = _.A[1][0] = _.A[2][0] = _.A[4][0] = _.A[5][0] = _.A[6][0] = zero;
				_.A[3][0] = identity;

				// Heptadignoal solver
				for (int i = 0; i < n; ++i) {
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
				for (int i = n - 1; i >= 0; --i) {
					_.x[i] = _.y[i];
					if (i + 1 < n)
						_.x[i] -= _.U[0][i] * _.x[i + 1];
					if (i + 2 < n)
						_.x[i] -= _.U[1][i] * _.x[i + 2];
					if (i + 3 < n)
						_.x[i] -= _.U[2][i] * _.x[i + 3];
				}

				// Assign back
				for (int i = 0; i < n; ++i)
					outVel[particleStartIndex + i] = _.x[i].segment<3>(0);
			});
		}

	HairEngine_Protected:
		IntermediateDataBuffer *dataBuffers = nullptr;
		int nDataBuffer = 0;
		int maxParticleSize = 0;
	};
}