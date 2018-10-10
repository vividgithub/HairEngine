
#ifdef HAIRENGINE_ENABLE_CUDA

#include "../util/helper_math.h"

namespace HairEngine {
	__global__
	void CudaSegmentMidpointComputer_cudaComputeMidpointKernal(const float3 *parPoses, const int *parStrandIndices,
	                                                           float3 *midpoints, int *segStrandIndices, int numParticle) {
		// Compute the midpoints, one for each segment
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= numParticle - 1)
			return;

		int si = parStrandIndices[i];
		if (si != parStrandIndices[i + 1])
			return;

		midpoints[i - si] = 0.5f * (parPoses[i] + parPoses[i + 1]);
		segStrandIndices[i - si] = parStrandIndices[i]; // Bypass the particle strand indices to segment
	}

	void CudaSegmentMidpointComputer_cudaComputeMidpoint(const float3 *parPoses, const int *parStrandIndices,
                                                     float3 *midpoints, int *segStrandIndices, int numParticle, int wrapSize) {

		int numThread = 32 * wrapSize;
		int numBlock = (numParticle + numThread - 1) / numThread;

		CudaSegmentMidpointComputer_cudaComputeMidpointKernal<<<numBlock, numThread>>>(parPoses,
				parStrandIndices, midpoints, segStrandIndices,  numParticle);
		cudaDeviceSynchronize();
	}
}

#endif
