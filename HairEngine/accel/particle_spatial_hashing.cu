
#include "../util/cuda_helper_math.h"
#include "../util/cudautil.h"
#include "../solver/hair_contacts_pbd_solver_lambda_computer.h"
#include <cstdio>

namespace HairEngine {

	__global__
	void ParticleSpatialHashing_computeHashValueByPositionsKernal(const float3 *poses, uint32_t *pHashes,
			float3 dInv, int numParticle, int shift) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= numParticle)
			return;

		pHashes[i] = getHashValueFromIndex3(make_int3(poses[i] * dInv), shift);
	}

	void ParticleSpatialHashing_computeHashValueByPositions(const float3 *poses, uint32_t *pHashes,
	                                                              float3 dInv, int numParticle, int shift, int wrapSize) {
		int numThread = 32 * wrapSize;
		int numBlock = (numParticle + numThread - 1) / numThread;

		ParticleSpatialHashing_computeHashValueByPositionsKernal<<<numBlock, numThread>>>(poses, pHashes, dInv, numParticle, shift);
		cudaDeviceSynchronize();
	}

	template <typename Func, typename RadiusProvider>
	__global__
	void ParticleSpatialHashing_rangeSearchKernel(
			Func func, // Pass by value to the kernel
			const int *hashStarts,
			const int *hashEnds,
			const int *pids,
			const float3 *positions,
			RadiusProvider radiusProvider,
			float3 dInv,
			int n,
			int hashShift
	) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= n)
			return;

		int pid1 = pids[idx];
		float3 pos1 = positions[pid1];

		func.before(pid1, pos1);

		float r = radiusProvider.radius(pid1, pos1);
		int3 index3Max = make_int3((pos1 + r) * dInv);
		int3 index3Min = make_int3((pos1 - r) * dInv);

		float r2 = r * r;
		for (int ix = index3Min.x; ix <= index3Max.x; ++ix)
			for (int iy = index3Min.y; iy <= index3Max.y; ++iy)
				for (int iz = index3Min.z; iz <= index3Max.z; ++iz) {
					uint32_t hash = getHashValueFromIndex3(make_int3(ix, iy, iz), hashShift);
					int hashStart = hashStarts[hash];
					if (hashStart == 0xffffffff)
						continue;
					int hashEnd = hashEnds[hash];

					for (int hashIt = hashStart; hashIt != hashEnd; ++hashIt) {
						int pid2 = pids[hashIt];
						float3 pos2 = positions[pid2];

						float distance2 = length2(pos2 - pos1);

						if (distance2 <= r2) {
							func(pid1, pid2, pos1, pos2, sqrtf(distance2));
						}
					}
				}

		func.after(pid1, pos1);
	}

	template <typename Func, typename RadiusProvider>
	void ParticleSpatialHashing_rangeSearch(
			const Func & func, // Pass by value to the kernel
			const int *hashStarts,
			const int *hashEnds,
			const int *pids,
			const float3 *positions,
			const RadiusProvider & radiusProvider,
			float3 dInv,
			int n,
			int hashShift,
			int wrapSize
	) {
		int numBlock, numThread;
		CudaUtility::getGridSizeForKernelComputation(n, wrapSize, &numBlock, &numThread);

		ParticleSpatialHashing_rangeSearchKernel<Func><<<numBlock, numThread>>>(func, hashStarts,
				hashEnds, pids, positions, radiusProvider, dInv, n, hashShift);
		cudaDeviceSynchronize();
	}

	struct ConstantRadiusProvider {
		float r;
		ConstantRadiusProvider(float r): r(r) {}

		__host__ __device__ __forceinline__
		float radius(int pid, float3 pos) const { return r; }
	};

	template <typename Func>
	void ParticleSpatialHashing_rangeSearch(
			const Func & func, // Pass by value to the kernel
			const int *hashStarts,
			const int *hashEnds,
			const int *pids,
			const float3 *positions,
			float r,
			float3 dInv,
			int n,
			int hashShift,
			int wrapSize
	) {
		ParticleSpatialHashing_rangeSearch<Func, ConstantRadiusProvider>(func, hashStarts,
				hashEnds, pids, positions, ConstantRadiusProvider(r), dInv, n, hashShift, wrapSize);
	}

	template <typename Func>
	void ParticleSpatialHashing_rangeSearch(
			const Func & func, // Pass by value to the kernel
			const int *hashStarts,
			const int *hashEnds,
			const int *pids,
			const float3 *positions,
			float3 dInv,
			int n,
			int hashShift,
			int wrapSize
	) {
		ParticleSpatialHashing_rangeSearch<Func, Func>(func, hashStarts,
				hashEnds, pids, positions, func, dInv, n, hashShift, wrapSize);
	}

	/*
	 * Register tempalte for Particle Spatial Hashing range search
	 */
	template void ParticleSpatialHashing_rangeSearch<HairContactsPBDDensityComputer>(
			const HairContactsPBDDensityComputer & func, // Pass by value to the kernel
			const int *hashStarts,
			const int *hashEnds,
			const int *pids,
			const float3 *positions,
			float r,
			float3 dInv,
			int n,
			int hashShift,
			int wrapSize
	);

	template void ParticleSpatialHashing_rangeSearch<HairContactsPBDPositionCorrectionComputer>(
			const HairContactsPBDPositionCorrectionComputer & func, // Pass by value to the kernel
			const int *hashStarts,
			const int *hashEnds,
			const int *pids,
			const float3 *positions,
			float r,
			float3 dInv,
			int n,
			int hashShift,
			int wrapSize
	);

	template void ParticleSpatialHashing_rangeSearch<HairContactsPBDViscosityComputer>(
			const HairContactsPBDViscosityComputer & func, // Pass by value to the kernel
			const int *hashStarts,
			const int *hashEnds,
			const int *pids,
			const float3 *positions,
			float r,
			float3 dInv,
			int n,
			int hashShift,
			int wrapSize
	);

	template void ParticleSpatialHashing_rangeSearch<HairContactsPBDCollisionFinder>(
			const HairContactsPBDCollisionFinder & func, // Pass by value to the kernel
			const int *hashStarts,
			const int *hashEnds,
			const int *pids,
			const float3 *positions,
			float3 dInv,
			int n,
			int hashShift,
			int wrapSize
	);
}