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

		HairContactsPBDDensityComputer(float *rhos, float3 *grad1, float *grad2, float *lambdas,
				int n, float h, float rho0, float f1, float f2):
				rhos(rhos), grad1(grad1), grad2(grad2), n(n), h(h), rho0(rho0), f1(f1), f2(f2) {}

		void clear() {
			cudaMemset(rhos, 0x00, sizeof(float) * n);
			cudaMemset(grad1, 0x00, sizeof(float3) * n);
			cudaMemset(grad2, 0x00, sizeof(float) * n);
		}

		__host__ __device__ __forceinline__
		void before(int pid1, float3 pos1) {}

		__host__ __device__ __forceinline__
		void operator()(int pid1, int pid2, float3 pos1, float3 pos2, float r) {
			// Compute the densities of pid1
			float g2 = (h - r);

			// Density
			float g1 = g2 * (h + r); // ( h^2 - r^2)
			g1 = g1 * g1 * g1; // (h^2 - r^2)^3
			rhos[pid1] += f1 * g1; // f1 * (h^2 - r^2)^3

			// Gradient, r = 0 gradient is not defined
			if (r > 0) {
				g2 = g2 * g2 * f2; // f2 * (h - r)^2
				grad1[pid1] += (pos2 - pos1) * (g2 / r);
				grad2[pid2] += g2 * g2; // Since the length of (pos2 - pos1) /r is 1
			}
		}

		__host__ __device__ __forceinline__
		void after(int pid1, float3 pos1) {
			lambdas[pid1] = (rho0 - rhos[pid1]) / (length2(grad1[pid1]) + grad2[pid1]);
		}

		float *rhos; ///< The output array of densities, MUST store in GPU
		float3 *grad1; ///< The grad1 in PBD Solver, MUST store in GPU
		float *grad2; ///< The grad2 in PBD Solver, MUST store in GPU
		float *lambdas; ///< The lambdas in PBD Solver, MUST store in GPU

		int n; ///< The size of particles
		float h; ///< The kernel radius
		float rho0; ///< The target density 0
		float f1; ///< Same as the f1 in PBD solver
		float f2; ///< Same as the f2 in PBD solver
	};

	struct HairContactsPBDPositionCorrectionComputer {

		HairContactsPBDPositionCorrectionComputer(const float *rhos, const float *lambdas,
		                           float3 *dxs, float3 *poses, float3 *vels, int n, int h, float t, float f2):
				rhos(rhos), lambdas(lambdas), dxs(dxs), poses(poses), vels(vels), n(n), h(h), tInv(1.0f / t), f2(f2) {}

		void clear() {
			cudaMemset(dxs, 0x00, sizeof(float3) * n);
		}

		__host__ __device__ __forceinline__
		void before(int pid1, float3 pos1) {}

		__host__ __device__ __forceinline__
		void operator()(int pid1, int pid2, float3 pos1, float3 pos2, float r) {
			// Gradient is not defined if r == 0
			if (r > 0) {
				float g2 = h - r;
				g2 = g2 * g2 * g2; // f2 * (h - r)^2
				dxs[pid1] += (pos2 - pos1) * (g2 * (lambdas[pid1] + lambdas[pid2]) / r); // (lambda[pid1] + lambda[pid2]) * gradient(W(i,j))
			}
		}

		__host__ __device__ __forceinline__
		void after(int pid1, float3 pos1) {
			// Compute the velocity
			float3 dx = dxs[pid1];
			poses[pid1] += dx;
			vels[pid1] = dx * tInv;
		}

		const float *rhos;
		const float *lambdas;
		float3 *dxs;
		float3 *poses;
		float3 *vels;

		int n;
		float h;
		float tInv; ///< The inverse of time
		float f2;
	};
}

#endif //HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_LAMBDA_COMPUTER_H
