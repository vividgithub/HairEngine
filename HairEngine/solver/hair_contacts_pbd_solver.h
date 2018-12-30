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

	void HairContactsPBDSolver_commitParticleVelocities(
			const float3 * poses2,
			const float3 * poses1,
			float3 *vels,
			float tInv,
			int numParticle,
			int wrapSize
	);

	void HairContactsPBDSolver_addCorrectionPositions(float3 *poses, const float3 *dxs, int n, int wrapSize);

	/**
	 * HairContactsPBDSolver uses position based dynamics to control the density
	 * of the hair particles. It provides similar effects as the impressible behaviors in fluid dynamics.
	 */
	class HairContactsPBDSolver: public Solver {

		friend class HairContactsPBDVisualizer;

		using DensityComputer = HairContactsPBDDensityComputer;
		using PositionCorrectionComputer = HairContactsPBDPositionCorrectionComputer;
		using ViscosityComputer = HairContactsPBDViscosityComputer;

	HairEngine_Public:

		HairContactsPBDSolver(
				float kernelRadius,
				float particleSize,
				float3 *currentPoses,
				const float3 *oldPoses,
				float3 *vels,
				float viscosityCoefficient = 0.05f,
				int numIteration = 2,
				float spatialHashingResolution = 1.0f,
				bool changeHairRoot = false,
				int wrapSize=8):
			numIteration(numIteration),
			h(kernelRadius),
			poses(currentPoses),
			oldPoses(oldPoses),
			vels(vels),
			vis(viscosityCoefficient),
			hSearch(kernelRadius / spatialHashingResolution),
			rho0(1.0f / (particleSize * particleSize * particleSize)),
			changeHairRoot(changeHairRoot),
			wrapSize(wrapSize) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {
			Solver::setup(hair, currentTransform);

			// Allocate space
			rhos = CudaUtility::allocateCudaMemory<float>(hair.nparticle);
			grad1 = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);
			grad2 = CudaUtility::allocateCudaMemory<float>(hair.nparticle);
			lambdas = CudaUtility::allocateCudaMemory<float>(hair.nparticle);
			dxs = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);
//			vels_ = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			// Allocate the density computer
			densityComputer = new DensityComputer(rhos, grad1, grad2, lambdas, hair.nparticle, h, rho0);
			positionCorrectionComputer = new PositionCorrectionComputer(rhos, lambdas, dxs, hair.nparticle, h, rho0);
//			viscosityComputer = new ViscosityComputer(vels_, vels, hair.nparticle, h, rho0, vis);

			psh = new ParticleSpatialHashing(hair.nparticle, make_float3(hSearch));
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			Solver::solve(hair, info);

			//Computer spatial hashing
			psh->update(poses, wrapSize);

			// Iterations
//			for (int _ = 0; _ < numIteration; ++_) {
//
//				auto t1 = std::chrono::high_resolution_clock::now();
//
//				psh->rangeSearch<DensityComputer>(*densityComputer, h, wrapSize);
//
//				auto t2 = std::chrono::high_resolution_clock::now();
//
//				psh->rangeSearch<PositionCorrectionComputer>(*positionCorrectionComputer, h, wrapSize);
//
//				auto t3 = std::chrono::high_resolution_clock::now();
//
//				// Add dxs to the poses buffer
//				HairContactsPBDSolver_addCorrectionPositions(poses, dxs, hair.nparticle, wrapSize);
//
//				auto tDensityCompute = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//				auto tVelocityProjection = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
//
//				printf("[HairContactsPBDSolver:DensityCompute] Timing %lld ms(%lld us)\n", tDensityCompute / 1000, tDensityCompute);
//				printf("[HairContactsPBDSolver:VelocityCorrect] Timing %lld ms(%lld us)\n", tVelocityProjection / 1000, tVelocityProjection);
//			}
//
//			// Compute the velocities
//			HairContactsPBDSolver_commitParticleVelocities(poses, oldPoses, vels, 1.0f / info.t, hair.nparticle, wrapSize);

			// Compute the velocity viscosity
//			psh->rangeSearch<ViscosityComputer>(*viscosityComputer, h, wrapSize);

			psh->rangeSearch(*densityComputer, h, wrapSize);
		}

		void tearDown() override {

			Solver::tearDown();

			delete psh;
			delete densityComputer;
//			delete viscosityComputer;

			CudaUtility::deallocateCudaMemory(rhos);
			CudaUtility::deallocateCudaMemory(grad1);
			CudaUtility::deallocateCudaMemory(grad2);
			CudaUtility::deallocateCudaMemory(lambdas);
			CudaUtility::deallocateCudaMemory(dxs);
//			CudaUtility::deallocateCudaMemory(vels_);
		}

	HairEngine_Protected:

		int numIteration; ///< Number of iterations per step

		float vis; ///< The visocity coefficient, used to create clumped and stiction effects for hairs

		float h; ///< The kernel radius
		float hSearch; ///< The radius used in spatial hashing search

		float rho0; ///< The target density, compute in the setup stage

		int wrapSize; ///< The cuda computation wrap size

		ParticleSpatialHashing *psh = nullptr;

		/// The sum of gradient, equals to \f$ \lvert \sum\limits_{k \neq i} \nabla_k W_{i,k} \rvert^2 $\f
		float3 *grad1 = nullptr;

		/// The sum of gradient square, equals to \f$ \sum\limits_{k \neq i} \lvert \nabla_k W_{i,k} \rvert^2 $\f
		float *grad2 = nullptr;

		/// The lambda \f$ \lambda_i $\f define in PBD solver
		float *lambdas = nullptr;

		float *rhos = nullptr; ///< The densities array, allocated in GPU

		float3 *poses; ///< The poses array, where computed positions will be stored in
		const float3 *oldPoses; ///< The old poses array, where the original postions are stored

		float3 *dxs; ///< The position correction values for each iterations

//		float3 *vels_; ///< An allocated space for storing the velocities before viscosity correction
		float3 *vels; ///< The velocities array

//		/// Equals to the particle that needs to be modified, if it needs to change the hair root, then n = hair.nparticle.
//		/// Else it equals to n = hair.nparticle - hair.nstrand
//		int n;
//
//		/// The offset for the particle that needs to modified. If changeHairRoot = true in the constructor, then
//		/// offset = hair.nstrand, otherwise offset = 0
//		int offset;

		bool changeHairRoot; ///< Whether to change the hair root

		DensityComputer *densityComputer = nullptr; ///< The density computer lambda
		PositionCorrectionComputer *positionCorrectionComputer = nullptr; ///< The position correction computer lambda
//		ViscosityComputer *viscosityComputer = nullptr; ///< The viscosity computer lambda
	};

	class HairContactsPBDVisualizer: public Visualizer {

	HairEngine_Public:

		HairContactsPBDVisualizer(const std::string &directory, const std::string &filenameTemplate, float timestep, HairContactsPBDSolver *solver)
				: solver(solver), Visualizer(directory, filenameTemplate, timestep) {}

		void visualize(std::ostream &os, Hair &hair, const IntegrationInfo &info) override {

			const int & n = hair.nparticle;

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


			float rho = 0.0f;
			for (int i = 0; i < n; ++i) {

				auto li1 = i / hair.nstrand;
				auto si1 = i % hair.nstrand;
				auto par = hair.strands[si1].particleInfo.beginPtr + li1;

				VPly::writePoint(
						os,
						EigenUtility::toVPlyVector3f(par->pos),
						VPly::VPlyVector3fAttr("g1", { grad1[i].x, grad1[i].y, grad1[i].z }),
						VPly::VPlyFloatAttr("g2", grad2[i]),
						VPly::VPlyFloatAttr("ld", lambdas[i]),
						VPly::VPlyFloatAttr("rho", rhos[i]),
						VPly::VPlyVector3fAttr("v", EigenUtility::toVPlyVector3f(par->vel)),
						VPly::VPlyVector3fAttr("dx", { dxs[i].x, dxs[i].y, dxs[i].z })
				);
				rho += rhos[i];
			}

			rho /= n;

			printf("[HairContactsPBDVisualizer]: rho=%e\n", rho);
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
		float3 *dxs = nullptr; ///< A copy of the solver->dxs in CPU, used to visualize
	};
}

#endif //HAIRENGINE_HAIR_CONTACTS_PBD_SOLVER_H
