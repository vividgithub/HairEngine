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
#include "visualizer.h"
#include "integration_info.h"

namespace HairEngine {

	class HairContactsPBDVisualizer;

	/**
	 * HairContactsPBDSolver uses position based dynamics to control the density
	 * of the hair particles. It provides similar effects as the impressible behaviors in fluid dynamics.
	 */
	class HairContactsPBDSolver: public CudaBasedSolver {

		friend class HairContactsPBDVisualizer;

	HairEngine_Protected:

		/**
		 * Lambda for computing density, grad1 and grad2, first range search in PBD solver
		 */
		struct DensityComputer {

			DensityComputer(float *rhos, float3 *grad1, float *grad2, float *lambdas, int n, float h, float rho0, float f1, float f2):
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

		struct PositionCorrectionComputer {

			PositionCorrectionComputer(const float *rhos, const float *lambdas,
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

	HairEngine_Public:

		HairContactsPBDSolver(float kernelRadius, int numIteration = 4, float targetDensityScale = 1.0f,
				float spatialHashingResolution = 1.0f, int wrapSize=8):
			CudaBasedSolver(Pos_ | Vel_),
			numIteration(numIteration),
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

			// Allocate space
			rhos = CudaUtility::allocateCudaMemory<float>(hair.nparticle);
			grad1 = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);
			grad2 = CudaUtility::allocateCudaMemory<float>(hair.nparticle);
			lambdas = CudaUtility::allocateCudaMemory<float>(hair.nparticle);
			dxs = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			// Allocate the density computer
			// Since rho0 is unknown and only relevant in computing lambdas, we can assign 0.0f to it first
			densityComputer = new DensityComputer(rhos, grad1, grad2, lambdas, hair.nparticle, 0.0f, h, f1, f2);

			psh = new ParticleSpatialHashing(hair.nparticle, make_float3(hSearch));
			psh->update(cmc->parRestPoses, wrapSize);

			// Compute the rest density
			densityComputer->clear();
			psh->rangeSearch<DensityComputer>(*densityComputer, h, wrapSize);
			rho0 = rho0Scale * thrust::reduce(thrust::device, rhos, rhos + hair.nparticle) / hair.nparticle;
			densityComputer->rho0 = rho0; // Rest the rho0
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			CudaBasedSolver::solve(hair, info);

			if (!positionCorrectionComputer)
				positionCorrectionComputer = new PositionCorrectionComputer(rhos, lambdas, dxs,
						cmc->parRestPoses, cmc->parVels, hair.nparticle, h, info.t, f2);

			//Computer spatial hashing
			psh->update(cmc->parPoses, wrapSize);

			// Position based iterations

			// With t = info.t / numIteration, so tInv = numIteration / info.t
			positionCorrectionComputer->tInv = numIteration / info.t;

			for (int _ = 0; _ < numIteration; ++_) {
				psh->rangeSearch<DensityComputer>(*densityComputer, h, wrapSize);
				psh->rangeSearch<PositionCorrectionComputer>(*positionCorrectionComputer, h, wrapSize);
			}
		}

		void tearDown() override {

			CudaBasedSolver::tearDown();

			delete psh;
			delete densityComputer;

			CudaUtility::deallocateCudaMemory(rhos);
			CudaUtility::deallocateCudaMemory(grad1);
			CudaUtility::deallocateCudaMemory(grad2);
			CudaUtility::deallocateCudaMemory(lambdas);
			CudaUtility::deallocateCudaMemory(dxs);
		}

	HairEngine_Protected:

		int numIteration; ///< Number of iterations per step

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

		/// The sum of gradient, equals to \f$ \lvert \sum\limits_{k \neq i} \nabla_k W_{i,k} \rvert^2 $\f
		float3 *grad1 = nullptr;

		/// The sum of gradient square, equals to \f$ \sum\limits_{k \neq i} \lvert \nabla_k W_{i,k} \rvert^2 $\f
		float *grad2 = nullptr;

		/// The lambda \f$ \lambda_i $\f define in PBD solver
		float *lambdas = nullptr;

		float *rhos = nullptr; ///< The densities array, allocated in GPU

		float3 *dxs = nullptr; ///< The dx array, where postion will be added

		DensityComputer *densityComputer = nullptr; ///< The density computer lambda
		PositionCorrectionComputer *positionCorrectionComputer = nullptr; ///< The position correction computer lambda

	};

	class HairContactsPBDVisualizer: public Visualizer {

	HairEngine_Public:

		HairContactsPBDVisualizer(HairContactsPBDSolver *solver, const std::string &directory,
				const std::string &filenameTemplate, float timestep)
				: solver(solver), Visualizer(directory, filenameTemplate, timestep) {}

		void visualize(std::ostream &os, Hair &hair, const IntegrationInfo &info) override {

			Visualizer::solve(hair, info);

			int n = hair.nparticle;

			// Allocate space if needed
			if (!grad1)
				grad1 = new float3[n];
			if (!grad2)
				grad2 = new float[n];
			if (!lambdas)
				lambdas = new float[n];
			if (!rhos)
				rhos = new float[n];
			if (!dxs)
				dxs = new float3[n];

			// Copy from device memory
			CudaUtility::copyFromDeviceToHost(grad1, solver->grad1, n);
			CudaUtility::copyFromDeviceToHost(grad2, solver->grad2, n);
			CudaUtility::copyFromDeviceToHost(lambdas, solver->lambdas, n);
			CudaUtility::copyFromDeviceToHost(rhos, solver->rhos, n);
			CudaUtility::copyFromDeviceToHost(dxs, solver->dxs, n);

			for (int i = 0; i < hair.nparticle; ++i) {
				auto par = hair.particles + i;
				VPly::writePoint(
						os,
						EigenUtility::toVPlyVector3f(par->pos),
						VPly::VPlyVector3fAttr("g1", { grad1[i].x, grad1[i].y, grad1[i].z }),
						VPly::VPlyFloatAttr("g2", grad2[i]),
						VPly::VPlyFloatAttr("ld", lambdas[i]),
						VPly::VPlyFloatAttr("rho", rhos[i]),
						VPly::VPlyVector3fAttr("dx", { dxs[i].x, dxs[i].y, dxs[i].z })
				);
			}
		}

		void tearDown() override {
			Visualizer::tearDown();

			delete [] grad1;
			delete [] grad2;
			delete [] lambdas;
			delete [] rhos;
			delete [] dxs;
		}

	HairEngine_Protected:

		HairContactsPBDSolver *solver;

		float3 *grad1 = nullptr; ///< A copy of the solver->grad1 in CPU, used to visualize
		float *grad2 = nullptr; ///< A copy of the solver->grad2 in CPU, used to visualize
		float *lambdas = nullptr; ///< A copy of the solver->lambdas in CPU, used to visualize
		float *rhos = nullptr; ///< A copy of the solver->rhos in CPU, used to visualize
		float3 *dxs = nullptr; ///< A copy of the solver->rhos in CPU, used to visualize
	};
}

#endif //HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_H
