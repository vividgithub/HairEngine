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

			// Make sure that all the strands has the same number of particles
			numParticleInStrand = hair.strands[0].particleInfo.nparticle;
			for (int i = 1; i < hair.nstrand; ++i) {
				if (hair.strands[i].particleInfo.nparticle != numParticleInStrand) {
					throw std::runtime_error("[CudaMemoryConverter] Different number of particles in strand for hair");
				}
			}

			numStrand = hair.nstrand;
			numParticle = hair.nparticle;

			parBufferHost = new float3[hair.nparticle];

			if (options & RestPos_) {
				parRestPoses = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);
				copyParticlePropertyToDeviceMemory<float3>(
						parRestPoses,
						[] (Hair::Particle::Ptr par) -> float3 {
							return EigenUtility::toFloat3(par->restPos);
						}
				);
			}

			if (options & Pos_)
				parPoses = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			if (options & Vel_)
				parVels = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			if (options & Impulse_)
				parImpulses = CudaUtility::allocateCudaMemory<float3>(hair.nparticle);

			if (options & LocalIndex_) {
				parLocalIndices = CudaUtility::allocateCudaMemory<unsigned char>(hair.nparticle);
				copyParticlePropertyToDeviceMemory<unsigned char>(
						parLocalIndices,
						[] (Hair::Particle::Ptr par) -> unsigned char {
							return static_cast<unsigned char>(par->localIndex);
						}
				);
			}

			if (options & GlobalIndex_) {
				parGlobalIndices = CudaUtility::allocateCudaMemory<int>(hair.nparticle);
				copyParticlePropertyToDeviceMemory<int>(
						parGlobalIndices,
						[] (Hair::Particle::Ptr par) -> int {
							return par->globalIndex;
						}
				);
			}

			if (options & StrandIndex_) {
				parStrandIndices = CudaUtility::allocateCudaMemory<int>(hair.nparticle);
				copyParticlePropertyToDeviceMemory<int>(
						parStrandIndices,
						[] (Hair::Particle::Ptr par) -> int {
							return par->strandIndex;
						}
				);
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
				copyParticlePropertyToDeviceMemory<float3>(
						parPoses,
						[] (Hair::Particle::Ptr par) -> float3 {
							return EigenUtility::toFloat3(par->pos);
						}
				);

			if (parVels)
				copyParticlePropertyToDeviceMemory<float3>(
						parVels,
						[] (Hair::Particle::Ptr par) -> float3 {
							return EigenUtility::toFloat3(par->vel);
						}
				);

			if (parImpulses)
				copyParticlePropertyToDeviceMemory<float3>(
						parImpulses,
						[] (Hair::Particle::Ptr par) -> float3 {
							return EigenUtility::toFloat3(par->impulse);
						}
				);
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

		float3 *parBufferHost = nullptr; ///< The host buffer for copying (float3)

		int numStrand = 0; ///< Number of strand in the hair
		int numParticleInStrand = 0; ///< Number of particle in the hair
		int numParticle = 0; ///< Number of particles

		int getCopyOptions() const {
			return options;
		}

	HairEngine_Protected:
		int options; ///< Which data should be copied

		template <typename T>
		void copyParticlePropertyToDeviceMemory(T *devicePtr, std::function<T(Hair::Particle::Ptr)> func) {
			T *hostPtr = reinterpret_cast<T*>(parBufferHost);

			for (int li = 0; li < numParticleInStrand; ++li)
				for (int si = 0; si < numStrand; ++si)
					hostPtr[li * numStrand + si] = func(hair->strands[si].particleInfo.beginPtr + li);

			CudaUtility::copyFromHostToDevice(devicePtr, hostPtr, numParticle);
		}
	};

	/**
	 * CudaMemoryInverseConverter converts the calculated device memory back to the hair's particle.
	 * A simple usage for the CudaMemoryConverter and CudaMemoryInverseConverter is to add it at the begin
	 * and end of several cuda-based hair solver, so the par position, vel, and impulse will be firsted copy
	 * to device memory and then convert back
	 */
	class CudaMemoryInverseConverter: public Solver {
	HairEngine_Public:
		CudaMemoryInverseConverter(CudaMemoryConverter * cudaMemoryConverter, int copyOptions = Pos_ | Vel_ | Impulse_):
			cudaMemoryConverter(cudaMemoryConverter), copyOptions(copyOptions) {}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			auto & cmc = cudaMemoryConverter; // Short name

			if (cmc->parPoses && (copyOptions & Pos_)) {
				CudaUtility::copyFromDeviceToHost(cmc->parBufferHost, cmc->parPoses, hair.nparticle);
				for (int si = 0; si < cmc->numStrand; ++si)
					for (int li = 0; li < cmc->numParticleInStrand; ++li) {
						Eigen::Vector3f & val = hair.strands[si].particleInfo.beginPtr[li].pos;
						val = EigenUtility::fromFloat3(cmc->parBufferHost[li * cmc->numStrand + si]);
					}
			}

			if (cmc->parVels && (copyOptions & Vel_)) {
				CudaUtility::copyFromDeviceToHost(cmc->parBufferHost, cmc->parVels, hair.nparticle);
				for (int si = 0; si < cmc->numStrand; ++si)
					for (int li = 0; li < cmc->numParticleInStrand; ++li) {
						Eigen::Vector3f & val = hair.strands[si].particleInfo.beginPtr[li].vel;
						val = EigenUtility::fromFloat3(cmc->parBufferHost[li * cmc->numStrand + si]);
					}
			}

			if (cmc->parImpulses && (copyOptions & Impulse_)) {
				CudaUtility::copyFromDeviceToHost(cmc->parBufferHost, cmc->parImpulses, hair.nparticle);
				for (int si = 0; si < cmc->numStrand; ++si)
					for (int li = 0; li < cmc->numParticleInStrand; ++li) {
						Eigen::Vector3f & val = hair.strands[si].particleInfo.beginPtr[li].impulse;
						val = EigenUtility::fromFloat3(cmc->parBufferHost[li * cmc->numStrand + si]);
					}
			}
		}

	HairEngine_Protected:
		CudaMemoryConverter *cudaMemoryConverter;
		int copyOptions;
	};
}

#endif
