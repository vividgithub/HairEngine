//
// Created by vivi on 2018/9/22.
//

#pragma once


#ifdef HAIRENGINE_ENABLE_CUDA

#include <cuda_runtime.h>
#include "../solver/solver.h"
#include "../util/cudautil.h"
#include "../util/eigenutil.h"

namespace HairEngine {

	enum CudaMemoryConverterOptions: int {
		RestPos_ = (1 << 0),
		Pos_ = (1 << 1),
		Vel_ = (1 << 2),
		Impulse_ = (1 << 3),
		LocalIndex_ = (1 << 4),
		GlobalIndex_ = (1 << 5),
		StrandIndex_ = (1 << 6)
	};

	/**
	 * CudaMemoryConverter copy the par->pos, par->vel and par->impulse to the
	 * cuda device memory.
	 */
	class CudaMemoryConverter: public Solver {

		friend class CudaMemoryInverseConverter;

	HairEngine_Public:

		CudaMemoryConverter(int options): options(options) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {

			parBufferHost = new float3[hair.nparticle];

			if (options & RestPos_) {
				parRestPoses = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);
				CudaUtility::copyFromHostToDevice(parRestPoses, hair.particles, hair.particles + hair.nparticle, parBufferHost,
						[] (const Hair::Particle & par) -> float3 { return EigenUtility::toFloat3(par.restPos); });
			}

			if (options & Pos_)
				parPoses = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			if (options & Vel_)
				parVels = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			if (options & Impulse_)
				parImpulses = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			if (options & LocalIndex_) {
				parLocalIndices = CudaUtility::allocateCudaMemory<unsigned char>(hair.nparticle);
				CudaUtility::copyFromHostToDevice(parLocalIndices, hair.particles, hair.particles + hair.nparticle, static_cast<unsigned char*>(nullptr),
						[] (const Hair::Particle & par) -> unsigned char { return static_cast<unsigned char>(par.localIndex); });
			}

			if (options & GlobalIndex_) {
				parGlobalIndices = CudaUtility::allocateCudaMemory<int>(hair.nparticle);
				CudaUtility::copyFromHostToDevice(parGlobalIndices, hair.particles, hair.particles + hair.nparticle, static_cast<int*>(nullptr),
					[] (const Hair::Particle & par) -> int { return par.globalIndex; });
			}

			if (options & StrandIndex_) {
				parStrandIndices = CudaUtility::allocateCudaMemory<int>(hair.nparticle);
				CudaUtility::copyFromHostToDevice(parStrandIndices, hair.particles, hair.particles + hair.nparticle, static_cast<int*>(nullptr),
				                                  [] (const Hair::Particle & par) -> int { return par.strandIndex; });
			}
		}

		void tearDown() override {
			CudaUtility::safeDeallocateCudaMemory(parPoses);
			CudaUtility::safeDeallocateCudaMemory(parVels);
			CudaUtility::safeDeallocateCudaMemory(parImpulses);
			delete[] parBufferHost;
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			if (parPoses)
				CudaUtility::copyFromHostToDevice(parPoses, hair.particles, hair.particles + hair.nparticle, parBufferHost,
						[] (const Hair::Particle & par) -> float3 { return EigenUtility::toFloat3(par.pos); });

			if (parVels)
				CudaUtility::copyFromHostToDevice(parVels, hair.particles, hair.particles + hair.nparticle, parBufferHost,
						[] (const Hair::Particle & par) -> float3 { return EigenUtility::toFloat3(par.vel); });

			if (parImpulses)
				CudaUtility::copyFromHostToDevice(parImpulses, hair.particles, hair.particles + hair.nparticle, parBufferHost,
						[] (const Hair::Particle & par) -> float3 { return EigenUtility::toFloat3(par.impulse); });
		}

	HairEngine_Public:
		float3 *parRestPoses = nullptr; ///< The null particle position buffer
		float3 *parPoses = nullptr; ///< The particle position buffer
		float3 *parVels = nullptr; ///< The particle velocities buffer
		float3 *parImpulses = nullptr; ///< The particle impulses buffer

		/// To increase the performace of cache (to check the whether it is hair root), we assume that the maximum
		/// discrete particle count for a single hair is not greater than 255
		unsigned char *parLocalIndices = nullptr;

		int *parGlobalIndices = nullptr; ///< The global index for the particles
		int *parStrandIndices = nullptr; ///< The strand index for the particles

		float3 *parBufferHost = nullptr; ///< The host buffer for copying

		int getCopyOptions() const {
			return options;
		}

	HairEngine_Protected:
		int options; ///< Which data should be copied
	};

	/**
	 * CudaMemoryInverseConverter converts the calculated device memory back to the hair's particle.
	 * A simple usage for the CudaMemoryConverter and CudaMemoryInverseConverter is to add it at the begin
	 * and end of several cuda-based hair solver, so the par position, vel, and impulse will be firsted copy
	 * to device memory and then convert back
	 */
	class CudaMemoryInverseConverter: public Solver {
	HairEngine_Public:
		CudaMemoryInverseConverter(CudaMemoryConverter * cudaMemoryConverter):
			cudaMemoryConverter(cudaMemoryConverter) {}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			auto & cmc = cudaMemoryConverter; // Short name

			if (cmc->parPoses) {
				CudaUtility::copyFromDeviceToHost(cmc->parBufferHost, cmc->parPoses, hair.nparticle);
				for (int i = 0; i < hair.nparticle; ++i)
					hair.particles[i].pos = EigenUtility::fromFloat3(cmc->parBufferHost[i]);
			}

			if (cmc->parVels) {
				CudaUtility::copyFromDeviceToHost(cmc->parBufferHost, cmc->parVels, hair.nparticle);
				for (int i = 0; i < hair.nparticle; ++i)
					hair.particles[i].vel = EigenUtility::fromFloat3(cmc->parBufferHost[i]);
			}

			if (cmc->parImpulses) {
				CudaUtility::copyFromDeviceToHost(cmc->parBufferHost, cmc->parImpulses, hair.nparticle);
				for (int i = 0; i < hair.nparticle; ++i)
					hair.particles[i].impulse = EigenUtility::fromFloat3(cmc->parBufferHost[i]);
			}
		}

	HairEngine_Protected:
		CudaMemoryConverter *cudaMemoryConverter;
	};
}

#endif
