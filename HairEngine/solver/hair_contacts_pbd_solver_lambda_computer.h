//
// Created by vivi on 2018/12/12.
//

/*
 * This header will try to decompose the DensityComputer and PositionCorrectionComputer from "hair_contacts_pbd_solver".
 * The reason for doing this is that we need to use DensityComputer and PositionCorrectionComputer in the "rangeSearch"
 * method in the ParticleSpatialHashing. Since the implementation is in cuda, including the big "hair_contacts_pbd_solver.h"
 * would need to include other unwanted header and could not compile it in cuda
 */

#ifndef HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_LAMBDA_COMPUTER_H
#define HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_LAMBDA_COMPUTER_H

#pragma once

#include "../util/cudautil.h"
#include "../util/cuda_helper_math.h"

namespace HairEngine {

	/**
	 * Lambda for computing density, grad1 and grad2, first range search in PBD solver
	 */
	struct HairContactsPBDDensityComputer {

		HairContactsPBDDensityComputer(
				float *rhos,
				float3 *grad1,
				float *grad2,
				float *lambdas,
				int n,
				float h,
				float rho0
		):
				rhos(rhos), grad1(grad1), grad2(grad2), lambdas(lambdas),
				n(n), h(h), rho0(rho0) {

			// Compute f1 and f2 based on h
			f1 = static_cast<float>(315.0 / (64.0 * M_PI * pow(h, 9.0)));
			f2 = static_cast<float>(-45.0 / (M_PI * pow(h, 6.0)));

		}

		__host__ __device__ __forceinline__
		void before(int pid1, float3 pos1) {
			rhos[pid1] = 0.0f;
			grad1[pid1] = make_float3(0.0f);
			grad2[pid1] = 0.0f;
		}

		__host__ __device__ __forceinline__
		void operator()(int pid1, int pid2, float3 pos1, float3 pos2, float r) {

			// Compute the densities of pid1
			// Density = (315 / (64 * pi * h^9)) * (h^2 - r^2)^3 = f1 * (h^2 - r^2)^3
			float g2 = (h - r); // h - r
			float g1 = g2 * (h + r); // ( h^2 - r^2)
			g1 = f1 * g1 * g1 * g1; // W(i, j)

			rhos[pid1] += g1; // f1 * (h^2 - r^2)^3

			// Gradient, r = 0 gradient is not defined
			// Gradient pid1 = (-45 / (pi * h^6)) * (h - r)^2 * (r / length(r))
			// Gradient pid2 = - Gradient i
			if (r > 0) {
				g2 = f2 * g2 * g2; // f2 * (h - r)^2
				// Add grad1 if pid2 is not fixed
				grad1[pid1] += (g2 / r) * (pos2 - pos1);
				// Add grad2 if pid1 is not fixed
				grad2[pid1] += g2 * g2;
			}
		}

		__host__ __device__ __forceinline__
		void after(int pid1, float3 pos1) {
			// Plus 1e-10f to avoid nan error
			lambdas[pid1] = - fmaxf(rhos[pid1] - rho0, 0.0f) / (length2(grad1[pid1]) + grad2[pid1] + 1e-10f);
		}

		float *rhos; ///< The output array of densities, MUST store in GPU
		float3 *grad1; ///< The grad1 in PBD Solver, MUST store in GPU
		float *grad2; ///< The grad2 in PBD Solver, MUST store in GPU
		float *lambdas; ///< The lambdas in PBD Solver, MUST store in GPU

		int n; ///< The size of particles
		float h; ///< The kernel radius
		float rho0; ///< The target density 0
		float f1; ///< The precomputed value for \f$W$\f computation in SPH, equals to \f$ \frac{315}{64 \pi h^9} $\f
		float f2; ///< The precomputed value for \f$\nabla W$\f computation in SPH, Spiky kernel, equals to \f$ -\frac{45}{\pi h^6}$\f
	};

	struct HairContactsPBDPositionCorrectionComputer {

		HairContactsPBDPositionCorrectionComputer(const float *rhos, const float *lambdas,
		                           float3 *dxs, int n, float h, float rho0):
				rhos(rhos),
				lambdas(lambdas),
				dxs(dxs),
				n(n),
				h(h),
				rho0(rho0) {

			// Compute f1, f2 and f3 based on h
			f1 = static_cast<float>(315.0 / (64.0 * M_PI * pow(h, 9.0)));
			f2 = static_cast<float>(-45.0 / (M_PI * pow(h, 6.0)));
		}

		__host__ __device__ __forceinline__
		void before(int pid1, float3 pos1) {
			dxs[pid1] = make_float3(0.0f);
		}

		__host__ __device__ __forceinline__
		void operator()(int pid1, int pid2, float3 pos1, float3 pos2, float r) {
			// Gradient is not defined if r == 0
			// Only compute dx when it is not fixed
			if (r > 0) {
				float g2 = h - r;
				dxs[pid1] += ((lambdas[pid1] + lambdas[pid2]) * f2 * g2 * g2 / r) * (pos1 - pos2);
			}
		}

		__host__ __device__ __forceinline__
		void after(int pid1, float3 pos1) {}

		const float *rhos;
		const float *lambdas;
		float3 *dxs;

		int n;
		float h;
		float rho0;
		float f1;
		float f2;
	};

	struct HairContactsPBDViscosityComputer {

		HairContactsPBDViscosityComputer(
				const float3 *vels_,
				float3 *vels,
				int n,
				float h,
				float rho0,
				float viscosity
		):
			vels_(vels_),
			vels(vels),
			n(n),
			h(h),
			volume0(1.0f / rho0),
			viscosity(viscosity),
			f1(static_cast<float>(315.0 / (64.0 * M_PI * pow(h, 9.0)))) {}

		__host__ __device__ __forceinline__
		void before(int pid1, float3 pos1) { vels[pid1] = (1 - viscosity) * vels_[pid1]; }

		__host__ __device__ __forceinline__
		void operator()(int pid1, int pid2, float3 pos1, float3 pos2, float r) {
			float g1 = (h - r) * (h + r); // ( h^2 - r^2)
			g1 = f1 * g1 * g1 * g1; // W(i, j)
			vels[pid1] += (viscosity * g1 * volume0) * vels_[pid2];
		}

		__host__ __device__ __forceinline__
		void after(int pid1, float3 pos1) {}

		const float3 *vels_;
		float3 *vels;
		int n;
		float h;
		float viscosity;
		float volume0;
		float f1;
	};
}

#endif //HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_LAMBDA_COMPUTER_H
