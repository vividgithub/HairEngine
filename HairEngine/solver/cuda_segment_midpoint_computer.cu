
#ifdef HAIRENGINE_ENABLE_CUDA

#include "../util/cuda_helper_math.h"

namespace HairEngine {
	__global__
	void CudaSegmentMidpointComputer_cudaComputeMidpointKernal(const float3 *parPoses, const int *parStrandIndices,
	                                                           float3 *midpoints, int *segStrandIndices,
	                                                           int numParticle, int numStrand) {
		// Compute the midpoints, one for each segment
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i + numStrand >= numParticle)
			return;

		float3 p1 = parPoses[i];
		float3 p2 = parPoses[i + numStrand];

		midpoints[i] = 0.5f * (p1 + p2);
//		lengths[i] = length(p2 - p1);
		segStrandIndices[i] = parStrandIndices[i]; // Bypass the particle strand indices to segment
	}

	void CudaSegmentMidpointComputer_cudaComputeMidpoint(const float3 *parPoses, const int *parStrandIndices,
                                                     float3 *midpoints, int *segStrandIndices, int numParticle, int numStrand, int wrapSize) {

		int numThread = 32 * wrapSize;
		int numBlock = (numParticle + numThread - 1) / numThread;

		CudaSegmentMidpointComputer_cudaComputeMidpointKernal<<<numBlock, numThread>>>(parPoses,
				parStrandIndices, midpoints, segStrandIndices, numParticle, numStrand);
		cudaDeviceSynchronize();
	}
}

#endif
