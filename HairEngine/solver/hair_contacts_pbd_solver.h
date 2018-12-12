//
// Created by vivi on 2018/12/11.
//

#ifndef HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_H
#define HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_H

#pragma once

#include <numeric>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include "../accel/particle_spatial_hashing.h"
#include "cuda_based_solver.h"
#include "visualizer.h"
#include "integration_info.h"
#include "hair_contacts_pbd_solver_lambda_computer.h"

namespace HairEngine {

	class HairContactsPBDVisualizer;

	/**
	 * HairContactsPBDSolver uses position based dynamics to control the density
	 * of the hair particles. It provides similar effects as the impressible behaviors in fluid dynamics.
	 */
	class HairContactsPBDSolver: public CudaBasedSolver {

		friend class HairContactsPBDVisualizer;

		using DensityComputer = HairContactsPBDDensityComputer;
		using PositionCorrectionComputer = HairContactsPBDPositionCorrectionComputer;

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

			{
				float *rhos_ = new float[hair.nparticle];
				CudaUtility::copyFromDeviceToHost(rhos_, rhos, hair.nparticle);
				rho0 = rho0Scale * std::accumulate(rhos_, rhos_ + hair.nparticle, 0.0f) / hair.nparticle;
				delete [] rhos_;
			}
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

		HairContactsPBDVisualizer(const std::string &directory, const std::string &filenameTemplate, float timestep, HairContactsPBDSolver *solver)
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
