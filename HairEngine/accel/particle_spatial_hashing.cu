
#include "../util/cuda_helper_math.h"

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
}