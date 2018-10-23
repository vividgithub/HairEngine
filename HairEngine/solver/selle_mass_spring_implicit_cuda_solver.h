//
// Created by vivi on 2018/10/22.
//

#pragma once

#include <HairEngine/HairEngine/geo/hair.h>
#include "cuda_based_solver.h"
#include "selle_mass_spring_solver_base.h"

#include "../util/simple_mat.h"

namespace HairEngine {

	void SelleMassSpringImplicitCudaSolver_resolveStrandDynamics(
			Mat3 **A,
			Mat3 **L,
			Mat3 **U,
			float3 *y,
			float3 *b,
			float3 *poses,
			float3 *prevPoses,
			float3 *restPoses,
			float3 *vels,
			float3 *impulses,
			const float *rigidness,
			Mat3 dTransform,
			float3 dTranslation,
			int numParticle,
			int numStrand,
			int numParticlePerStrand,
			float pmass,
			float damping,
			float kStretch,
			float kBending,
			float kTorsion,
			float strainLimitingTolerance,
			float t,
			int wrapSize
	);

	void SelleMassSpringImplicitCudaSolver_getVelocityFromPosition(
			const float3 * poses,
			const float3 * prevPoses,
			float3 * vels,
			float tInv,
			int numParticle,
			int wrapSize
	);

	class SelleMassSpringImplicitCudaSolver: public CudaBasedSolver {

	HairEngine_Public:

		using Configuration = SelleMassSpringSolverBase::Configuration;

		SelleMassSpringImplicitCudaSolver(const Configuration & conf, int wrapSize = 8):
			CudaBasedSolver(Pos_ | RestPos_ | Vel_ | Impulse_),
			conf(conf), wrapSize(wrapSize) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {
			CudaBasedSolver::setup(hair, currentTransform);

			for (int i = 0; i < 7; ++i)
				A[i] = CudaUtility::allocateCudaMemory<Mat3>(hair.nparticle);
			for (int i = 0; i < 5; ++i)
				L[i] = CudaUtility::allocateCudaMemory<Mat3>(hair.nparticle);
			for (int i = 0; i < 3; ++i)
				U[i] = CudaUtility::allocateCudaMemory<Mat3>(hair.nparticle);

			b = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);
			y = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			prevPoses = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);
			poses = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			// Get the average length and compute the rigidness
			float averageLength = 0.0f;
			float *rigidnessHost = new float[hair.nparticle];

			for (int si = 0; si < hair.nstrand; ++si) {
				auto & st = hair.strands[si];

				// Compute the strand length
				float strandLength = 0.0f;
				for (auto seg(st.segmentInfo.beginPtr); seg != st.segmentInfo.endPtr; ++seg)
					strandLength += (seg->p2->restPos - seg->p1->restPos).norm();

				averageLength += strandLength;

				// Assign the rigidness to each particle
				float currentLength = 0.0f;
				for (int i = 0; i < st.particleInfo.nparticle; ++i) {
					rigidnessHost[i * hair.nstrand + si] = conf.rigidness(currentLength / strandLength);
					if (i + 1 < st.particleInfo.nparticle)
						currentLength += (st.particleInfo.beginPtr[i + 1].restPos - st.particleInfo.beginPtr[i].restPos).norm();
				}
			}

			CudaUtility::copyFromHostToDevice(rigidness, rigidnessHost, hair.nparticle);
			delete [] rigidnessHost;

			averageLength /= hair.nstrand;

			// Replace conf.mass to pmass in the configuraiton
			// Replace strand stiffness to particle stiffness
			conf.mass *= static_cast<float>(hair.nstrand) / static_cast<float>(hair.nparticle);
			conf.stretchStiffness /= averageLength;
			conf.bendingStiffness /= averageLength;
			conf.torsionStiffness /= averageLength;
			conf.altitudeStiffness /= averageLength;
		}

		void tearDown() override {
			for (int i = 0; i < 7; ++i)
				CudaUtility::deallocateCudaMemory(A[i]);
			for (int i = 0; i < 5; ++i)
				CudaUtility::deallocateCudaMemory(L[i]);
			for (int i = 0; i < 3; ++i)
				CudaUtility::deallocateCudaMemory(U[i]);
			CudaUtility::deallocateCudaMemory(b);
			CudaUtility::deallocateCudaMemory(y);

			CudaUtility::deallocateCudaMemory(prevPoses);
			CudaUtility::deallocateCudaMemory(poses);
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {

			std::vector<float> timeIntervals = { 0.0f };
			float tinc = (conf.maxIntegrationTime <= 0.0f) ? 1.0f : conf.maxIntegrationTime / info.t;
			while (timeIntervals.back() + tinc <= 1.0f) {
				timeIntervals.push_back(timeIntervals.back() + tinc);
			}
			if (timeIntervals.back() <= 0.995f)
				timeIntervals.push_back(1.0f);

			auto splittedInfos = info.lerp(timeIntervals.cbegin(), timeIntervals.cend());

			for (int i = 0; i < splittedInfos.size(); ++i) {
				const auto & splittedInfo = splittedInfos[i];

				auto dTransform_ = splittedInfo.tr * splittedInfo.ptr.inverse(Eigen::Affine);
				const auto & dTransformMat = dTransform_.matrix();

				Mat3 dTransform(
						dTransformMat(0, 0), dTransformMat(0, 1), dTransformMat(0, 2),
						dTransformMat(1, 0), dTransformMat(1, 1), dTransformMat(1, 2),
						dTransformMat(2, 0), dTransformMat(2, 1), dTransformMat(2, 2)
				);

				float3 dTranslation {
					dTransformMat(0, 3),
					dTransformMat(1, 3),
					dTransformMat(2, 3)
				};

				// Use (i == 0) ? cmc->parPoses : prevPoses to avoid an extra copy
				SelleMassSpringImplicitCudaSolver_resolveStrandDynamics(
						(Mat3**)A, (Mat3**)L, (Mat3**)U, y, b, poses, (i == 0) ? cmc->parPoses : prevPoses, cmc->parRestPoses,
						cmc->parVels, cmc->parImpulses, rigidness, dTransform, dTranslation, hair.nparticle,
						hair.nstrand, cmc->numParticleInStrand, conf.mass, conf.damping, conf.stretchStiffness,
						conf.bendingStiffness, conf.torsionStiffness, conf.strainLimitingLengthTolerance, splittedInfo.t,
						wrapSize
				);

				// For next iteration
				std::swap(poses, prevPoses);
			}

			// Compute the velocity, now the final pos result is stored in "prevPoses" (after swapping) and the
			// old result before iteration is stored in "cmc->parPoses".
 			SelleMassSpringImplicitCudaSolver_getVelocityFromPosition(
					prevPoses, cmc->parPoses, cmc->parVels, 1.0f / info.t, hair.nparticle, wrapSize
			);

			// Particle not commit, so we don't need to apply an update to cmc->parPoses
		}

	HairEngine_Protected:
		Configuration conf;

		Mat3 *A[7];
		Mat3 *L[5];
		Mat3 *U[3];

		float3 *b;
		float3 *y;

		float3 *prevPoses;
		float3 *poses;

		float *rigidness;

		int wrapSize;
	};
}
