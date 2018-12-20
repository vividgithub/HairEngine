#include "../accel/particle_spatial_hashing.h"
#include "../util/cudautil.h"
#include "../util/cuda_helper_math.h"

namespace HairEngine {

	__global__
	void HairContactsPBDSolver_addCorrectionPositionsKernel(float3 *poses, const float3 *dxs, int n) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= n)
			return;

		poses[i] += dxs[i];
	}

	void HairContactsPBDSolver_addCorrectionPositions(float3 *poses, const float3 *dxs, int n, int wrapSize) {
		int numBlock, numThread;
		CudaUtility::getGridSizeForKernelComputation(n, wrapSize, &numBlock, &numThread);

		HairContactsPBDSolver_addCorrectionPositionsKernel<<<numBlock, numThread>>>(poses, dxs, n);
		cudaDeviceSynchronize();
	}

	__global__
	void HairContactsPBDSolver_commitParticleVelocitiesKernel(
			const float3 * poses2,
			const float3 * poses1,
			float3 *vels,
			float tInv,
			int numParticle) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		// We don't change the root velocity
		if (i >= numParticle)
			return;

		vels[i] = (poses2[i] - poses1[i]) * tInv;
	}

	void HairContactsPBDSolver_commitParticleVelocities(
			const float3 * poses2,
			const float3 * poses1,
			float3 *vels,
			float tInv,
			int numParticle,
			int wrapSize) {

		int numBlock, numThread;
		CudaUtility::getGridSizeForKernelComputation(numParticle, wrapSize, &numBlock, &numThread);

		HairContactsPBDSolver_commitParticleVelocitiesKernel<<<numBlock, numThread>>>(poses2, poses1, vels,
				tInv, numParticle);
		cudaDeviceSynchronize();
	}
}