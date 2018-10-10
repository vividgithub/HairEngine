//
// Created by vivi on 2018/10/5.
//

#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include "../util/helper_math.h"
#include "../util/cudautil.h"
#include "../util/parallutil.h"
#include "../precompiled/precompiled.h"

namespace HairEngine {

	void ParticleSpatialHashing_computeHashValueByPositions(const float3 *poses, uint32_t *pHashes, float3 dInv,
			int numParticle, int shift, int wrapSize);

	/**
	 * ParticleSpatialHashing is a cuda-based implementation of "Optimized Spatial Hashing for Collision
	 * Detection of Deformable Objects" which builds a spatial grid for particles.
	 * It's suitable for fixed range and nearest neighbour search. For more detail, see the original paper.
	 */
	class ParticleSpatialHashing final {

	HairEngine_Private:
		int radixStarts[1 << 16]; ///< The count and start index buffer for radix sort

		int *hashParStarts = nullptr; ///< The particle start index for the hash entries, size equals to numHash
		int *hashParEnds = nullptr; ///< The particle end index for the hash entries, size equals to numHash

	HairEngine_Public:
		int *hashParStartsDevice = nullptr; ///< The device copy of "hashParStarts" in GPU
		int *hashParEndsDevice = nullptr; ///< The device copy of "hashParEnds" in GPU

	HairEngine_Private:
		int *pids = nullptr; ///< The particle ids (in CPU), used for radix sort
		int *pidsSwap = nullptr; ///< The swap area for the "pids", used in radix sort
		uint32_t *pHashes = nullptr; ///< The particle hashes value in (CPU)
		uint32_t *pHashesSwap = nullptr; ///< The swap area for the "pHashes", used in radix sort

	HairEngine_Public:
		int *pidsDevice = nullptr; ///< The "pids" copy in the GPU
		uint32_t *pHashesDevice = nullptr; ///< The particle hashes value in (GPU)

		int numParticle; ///< The number of particles
		int numHash; ///< The number of hash entries

		/// In order to compute the hash value fast without introducing "mod", we assume that the numHash
		/// is always the power of 2, means than numHash = 2^(numHashShift), so the mod operation can be replaced
		/// by >>(shift)
		int numHashShift;

		float3 d; ///< The diagonal of the voxel
		float3 dInv; ///< The element-wise inverse of "d"

	HairEngine_Public:

		/**
		 * Allocate space for the structure
		 * @param numParticle Number of particles for allocating
		 * @param d The diagonal of the voxel size
		 * @param loadBalance The load balance factor, which the number of hash entry will be equal to
		 * the smallest 2^n, where 2^n >= loadBalance * nparticle
		 */
		ParticleSpatialHashing(int numParticle, float3 d, float loadBalance = 1.0f):
			numParticle(numParticle),
			d(d),
			numHash(static_cast<int>(loadBalance * numParticle)) {

			// Make "numHash" to the smallest 2^n where 2^n larger than current value of "numHash"
			numHashShift = 0;
			while ((1 << numHashShift) < numHash) {
				++numHashShift;
			}
			numHash = (1 << numHashShift);

			dInv = float3 { 1.0f / d.x, 1.0f / d.y, 1.0f / d.z };

			hashParStarts = new int[numHash];
			hashParEnds = new int[numHash];
			hashParStartsDevice = CudaUtility::allocateCudaMemory<int>(numHash);
			hashParEndsDevice = CudaUtility::allocateCudaMemory<int>(numHash);

			pids = new int[numParticle];
			pidsSwap = new int[numParticle];
			pidsDevice = CudaUtility::allocateCudaMemory<int>(numParticle);

			pHashes = new uint32_t[numParticle];
			pHashesSwap = new uint32_t[numParticle];
			pHashesDevice = CudaUtility::allocateCudaMemory<uint32_t>(numParticle);
		}

		~ParticleSpatialHashing() {
			delete[] hashParStarts;
			delete[] hashParEnds;
			CudaUtility::deallocateCudaMemory(hashParStartsDevice);
			CudaUtility::deallocateCudaMemory(hashParEndsDevice);

			delete[] pids;
			delete[] pidsSwap;
			CudaUtility::deallocateCudaMemory(pidsDevice);

			delete[] pHashes;
			delete[] pHashesSwap;
			CudaUtility::deallocateCudaMemory(pHashesDevice);
		}

		/**
		 * Update the particle spatial hash structure from a group of particles
		 * @param poses The particle positions array, we assume that the "poses" are allocated in GPU memory and
		 * its size equals to "numParticle"
		 *
		 * @param poses The particle position array
		 * @param wrapSize The wrapSize for a thread block to execute in cuda
		 */
		void update(const float3 *poses, int wrapSize = 8) {
			// Compute the hash value in GPU
			ParticleSpatialHashing_computeHashValueByPositions(poses, pHashesDevice, dInv, numParticle, numHashShift, wrapSize);

			// Copy the computed hash value from device to host
			CudaUtility::copyFromDeviceToHost(pHashes, pHashesDevice, numParticle);

			// Assign the "pids", used in radix sort
			for (int i = 0; i < numParticle; ++i)
				pids[i] = i;

			// Radix sort, 2 runs, 1 run for 16-bit, total 32-bit
			for (int run = 0; run < 2; ++run) {
				int runShift = 16 * run;

				// Fill to zero
				memset(radixStarts, 0, sizeof(int) * 65536);

				// Count the radix
				for (int i = 0; i < numParticle; ++i) {
					uint32_t offset = (pHashes[i] >> runShift) & 65535u;
					++radixStarts[offset];
				}

				// Scan to get the start index
				int sum = 0;
				for (int i = 0; i < 65536; ++i) {
					int tmp = radixStarts[i];
					radixStarts[i] = sum;
					sum += tmp;
				}

				// Place to the swap buffer
				for (int i = 0; i < numParticle; ++i) {
					uint32_t offset = (pHashes[i] >> runShift) & 65535u;

					int swapIdx = radixStarts[offset]++;
					pHashesSwap[swapIdx] = pHashes[i];
					pidsSwap[swapIdx] = pids[i];
				}

				// For next iterations
				std::swap(pHashes, pHashesSwap);
				std::swap(pids, pidsSwap);
			}

			// Get the start index and end index for the hash entries
			// Don't need to fill the "hashParEnds", only need to check whether hashParStarts is 0xffffffff
			memset(hashParStarts, 0xff, sizeof(int) * numHash);

			// Scan to get the start index and end index
			for (int s = 0, t = 0; s < numParticle;) {
				t = s;
				while (t < numParticle && pHashes[t] == pHashes[s])
					++t;

				hashParStarts[pHashes[s]] = s;
				hashParEnds[pHashes[s]] = t;

				s = t;
			}

			// Copy to the cuda memory buffer for "pids", "hashParStarts" and "hashParEnds"
			CudaUtility::copyFromHostToDevice(pidsDevice, pids, numParticle);
			CudaUtility::copyFromHostToDevice(hashParStartsDevice, hashParStarts, numHash);
			CudaUtility::copyFromHostToDevice(hashParEndsDevice, hashParEnds, numHash);
		}
	};
}
