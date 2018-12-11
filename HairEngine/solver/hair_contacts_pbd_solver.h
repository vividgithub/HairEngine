//
// Created by vivi on 2018/12/11.
//

#ifndef HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_H
#define HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_H

#pragma once

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "../accel/particle_spatial_hashing.h"
#include "cuda_based_solver.h"
#include "integration_info.h"

namespace HairEngine {

	float computeDensity(
			const int *hashStarts,
			const int *hashEnds,
			const int *pids,
			const float3 *poses,
			float *outDensities,
			int n,
			float h,
			int hashShift,
			int wrapSize
	);

	/**
	 * HairContactsPBDSolver uses position based dynamics to control the density
	 * of the hair particles. It provides similar effects as the impressible behaviors in fluid dynamics.
	 */
	class HairContactsPBDSolver: public CudaBasedSolver {

	HairEngine_Public:

		HairContactsPBDSolver(float kernelRadius, float targetDensityScale = 1.0f,
				float spatialHashingResolution = 1.0f, int wrapSize=8):
			CudaBasedSolver(Pos_ | Vel_),
			h(kernelRadius),
			hSearch(kernelRadius / spatialHashingResolution),
			rho0Scale(targetDensityScale),
			wrapSize(wrapSize) {

			// Compute f1 and f2 based on h
			f1 = static_cast<float>(315.0 / (64.0 * M_PI * pow(h, 9.0)));
			f2 = static_cast<float>(-45.0 / (M_PI) * pow(h, 6.0));

		}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {
			CudaBasedSolver::setup(hair, currentTransform);

			lambdas = CudaUtility::allocateCudaMemory<float>(hair.nparticle);

			psh = new ParticleSpatialHashing(hair.nparticle, make_float3(hSearch));

			psh->update(cmc->parRestPoses, wrapSize);

			// Compute the rest density
			auto & densities = lambdas; // We store the densities in the lambdas array
			computeDensity(
					psh->hashParStartsDevice,
					psh->hashParEndsDevice,
					psh->pidsDevice,
					cmc->parRestPoses,
					densities,
					hair.nparticle,
					h,
					psh->numHashShift,
					wrapSize
			);
			rho0 = rho0Scale * thrust::reduce(thrust::device, densities, densities + hair.nparticle) / hair.nparticle;
		}

		void tearDown() override {
			delete psh;
		}

	HairEngine_Protected:

		float h; ///< The kernel radius
		float hSearch; ///< The radius used in spatial hashing search
		float f1; ///< The precomputed value for \f$W$\f computation in SPH, equals to \f$ \frac{315}{64 \pi h^9} $\f
		float f2; ///< The precomputed value for \f$\nabla W$\f computation in SPH, Spiky kernel, equals to \f$ -\frac{45}{\pi h^6}$\f

		float rho0; ///< The target density, compute in the setup stage

		/// The target density scale, we compute the target density based on the rest geometry. This parameter can
		/// use as a multiplier to the computed density
		float rho0Scale;

		int wrapSize; ///< The cuda computation wrap size

		ParticleSpatialHashing *psh = nullptr;
		float *lambdas = nullptr;
	};
}

#endif //HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_H
