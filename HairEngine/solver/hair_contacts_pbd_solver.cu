#include "../accel/particle_spatial_hashing.h"
#include "../util/cudautil.h"
#include "../util/cuda_helper_math.h"
#include "hair_contacts_pbd_solver_lambda_computer.h"

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
	void HairContactsPBDSolver_computeMidpointsKernel(const float3 *poses, float3 *midpoints, int numSegment, int numStrand) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= numSegment)
			return;
		midpoints[i] = (poses[i] + poses[i + numStrand]) * 0.5f;
	}

	void HairContactsPBDSolver_computeMidpoints(const float3 *poses, float3 *midpoints, int numSegment, int numStrand, int wrapSize) {
		int numBlock, numThread;
		CudaUtility::getGridSizeForKernelComputation(numSegment, wrapSize, &numBlock, &numThread);

		HairContactsPBDSolver_computeMidpointsKernel<<<numBlock, numThread>>>(poses, midpoints, numSegment, numStrand);
		cudaDeviceSynchronize();
	}

	__global__
	void HairContactsPBDSolver_computeCorrectionForCollisionsKernel(const int *numPairs,
			const HairContactsPBDCollisionPairInfo *pairs, const float3 *poses, float3 *dxs, float l,
			int numSegment, int numStrand) {

		int sid1 = blockDim.x * blockIdx.x + threadIdx.x;
		if (sid1 >= numSegment)
			return;

		float3 p1 = poses[sid1];
		float3 d1 = poses[sid1 + numStrand] - p1;

		auto itStart = pairs + sid1;
		auto itEnd = itStart + numPairs[sid1] * numSegment;
		for (auto pair = itStart; pair != itEnd; pair += numSegment) {

			const auto & sid2 = pair->index;
			const auto & s = pair->cpa.x;
			const auto & t = pair->cpa.y;
			const auto & n = pair->n;
			const auto & tau = pair->tau;

//			if (sid1 == 77)
//				printf("%d(%d) ---> %d: cpa={%f, %f}, n={%f, %f, %f}, tau=%f}\n", sid1, numPairs[sid1], sid2, s, t, n.x, n.y, n.z, tau);

			float3 p2 = poses[sid2];
			float3 d2 = poses[sid2 + numStrand] - p2;

			float lambda = tau * (dot((p1 + s * d1) - (p2 + t * d2), n) - l);
			float3 n_ = -lambda * n;

//			if (sid1 == 77) {
//				printf("%d ---> %d: p1={%f, %f, %f}, d1={%f, %f, %f}, p2={%f, %f, %f}, d2={%f, %f, %f}\n", sid1, sid2, p1.x, p1.y, p1.z, d1.x, d1.y, d1.z, p2.x, p2.y, p2.z, d2.x, d2.y, d2.z);
//				printf("%d ---> %d: lambda=%f, n_={%f, %f, %f}\n", sid1, sid2, lambda, n_.x, n_.y, n_.z);
//				printf("%d ---> %d: theta1=%f, theta2=%f\n", sid1, sid2, 180.0f / M_PI * acosf(abs(dot(n, d1 / length(d1)))), 180.0f / M_PI * acosf(abs(dot(n, d2 / length(d2)))));
//			}


			// Write for sid1
			atomicAdd(dxs + sid1, (1 - s) * n_);
			atomicAdd(dxs + sid1 + numStrand, s * n_);
			atomicAdd(dxs + sid2, (t - 1) * n_);
			atomicAdd(dxs + sid2 + numStrand, -t * n_);
		}
	}

	void HairContactsPBDSolver_computeCorrectionForCollisions(const int *numPairs,
			const HairContactsPBDCollisionPairInfo *pairs, const float3 *poses, float3 *dxs, float l,
			int numSegment, int numStrand, int wrapSize) {
		int numBlock, numThread;
		CudaUtility::getGridSizeForKernelComputation(numSegment, wrapSize, &numBlock, &numThread);

		HairContactsPBDSolver_computeCorrectionForCollisionsKernel<<<numBlock, numThread>>>(numPairs, pairs,
				poses, dxs, l, numSegment, numStrand);
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