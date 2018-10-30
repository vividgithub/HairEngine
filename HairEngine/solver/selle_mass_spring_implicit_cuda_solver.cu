
#include "cuda_runtime.h"
#include "../util/simple_mat.h"

namespace HairEngine {

	__device__ __forceinline__
	void getSpringInfo(float3 p1, float3 p2, float3 rp1, float3 rp2, float t, float k, float3 &impulse, Mat3 &dm) {
		float3 d = p2 - p1;
		float l = length(d);
		d /= l;

		float f = t * t * k;

		impulse = (k * (l - length(rp1 - rp2)) * t) * d;

		dm.at<0, 0>() = f * d.x * d.x;
		dm.at<0, 1>() = dm.at<1, 0>() = f * d.x * d.y;
		dm.at<0, 2>() = dm.at<2, 0>() = f * d.x * d.z;
		dm.at<1, 1>() = f * d.y * d.y;
		dm.at<1, 2>() = dm.at<2, 1>() = f * d.y * d.z;
		dm.at<2, 2>() = f * d.z * d.z;
	}

	__global__
	void SelleMassSpringImplicitCudaSolver_resolveStrandDynamicsKernal(
			Mat3 *L0, Mat3 *L1, Mat3 *L2, Mat3 *L3,
			Mat3 *U0, Mat3 *U1, Mat3 *U2,
			float3 *poses,
			float3 *prevPoses,
			float3 *restPoses,
			float3 *vels,
			float3 *impulses,
			const float *rigidness,
			Mat3 dTransform,
			float3 dTranslation,
			int numParticle,
			int numStrand,
			int numParticlePerStrand,
			float pmass,
			float damping,
			float kStretch,
			float kBending,
			float kTorsion,
			float strainLimitingTolerance,
			float t
	) {
		extern __shared__ int shared[];

		// One for each strand
		int si = blockIdx.x * blockDim.x + threadIdx.x;
		if (si >= numStrand)
			return;

		// Alias
		const auto & n = numParticlePerStrand;
		auto & b = vels;
		char* buffer = (char*)(shared) + threadIdx.x *4 * sizeof(Mat3);

		Mat3 & dm = ((Mat3*)buffer)[0];
		float3 *p = (float3*)(&dm + 1);
		float3 *rp = p + 4;

		float3 impulse;
		for (int i = si, li = 0; i < numParticle; i += numStrand, ++li) {

			L0[i].asDiagonal(pmass + damping * t);
			L3[i].asZero(); L2[i].asZero(); L1[i].asZero();
			U0[i].asZero(); U1[i].asZero(); U2[i].asZero();

			b[i] = pmass * vels[i] + impulses[i] * t;

			// Apply spring force, since b[pi] and A[xxx][pi] has been correctly initialized, it is safe to modify
			p[3] = prevPoses[i];
			rp[3] = restPoses[i];

			// Stretch spring
			if (li >= 1) {
				getSpringInfo(p[2], p[3], rp[2], rp[3], t, kStretch, impulse, dm);

				auto pi = i - numStrand;
				b[pi] += impulse;
				b[i] -= impulse;

				L0[pi] += dm;
				L0[i] += dm;
				U0[pi] -= dm;
				L1[i] -= dm;
			}

			// Bending spring
			if (li >= 2) {
				getSpringInfo(p[1], p[3], rp[1], rp[3], t, kBending, impulse, dm);

				auto pi = i - 2 * numStrand;
				b[pi] += impulse;
				b[i] -= impulse;

				L0[pi] += dm;
				L0[i] += dm;
				U1[pi] -= dm;
				L2[i] -= dm;
			}

			// Torsion spring
			if (li >= 3) {
				getSpringInfo(p[0], p[3], rp[0], rp[3], t, kTorsion, impulse, dm);

				auto pi = i - 3 * numStrand;
				b[pi] += impulse;
				b[i] -= impulse;

				L0[pi] += dm;
				L0[i] += dm;
				U2[pi] -= dm;
				L3[i] -= dm;
			}

			p[0] = p[1]; p[1] = p[2]; p[2] = p[3];
			rp[0] = rp[1]; rp[1] = rp[2]; rp[2] = rp[3];
		}

		// Initialize b and A of the strand root
		b[si] = (dTransform * prevPoses[si] + dTranslation - prevPoses[si]) / t;
		L3[si].asZero(); L2[si].asZero(); L1[si].asZero();
		U0[si].asZero(); U1[si].asZero(); U2[si].asZero();
		L0[si].asIdentity();

		// Heptadignoal solver
		Mat3 & L0_ = ((Mat3*)buffer)[0];
		Mat3 & L1_ = (&L0_)[1];
		Mat3 & L2_ = (&L1_)[1];
		Mat3 & L3_ = (&L2_)[1];

		int i1, i2, i3;
		for (int i = si, li = 0; i < numParticle; i += numStrand, ++li) {

			// Load
			i1 = i - numStrand;
			i2 = i1 - numStrand;
			i3 = i2 - numStrand;

			L0_ = L0[i];
			L1_ = L1[i];
			L2_ = L2[i];
			L3_ = L3[i];

			if (li >= 3) {
				L2_ -= L3_ * U0[i3];
				L1_ -= L3_ * U1[i3];
				L0_ -= L3_ * U2[i3];
				b[i] -= L3_ * b[i3];
			}

			if (li >= 2) {
				L1_ -= L2_ * U0[i2];
				L0_ -= L2_ * U1[i2];
				U0[i] -= L2_ * U2[i2];
				b[i] -= L2_ * b[i2];
			}

			if (li >= 1) {
				L0_ -= L1_ * U0[i1];
				U1[i] -= L1_ * U2[i1];
				U0[i] -= L1_ * U1[i1];
				b[i] -= L1_ * b[i1];
			}

			L0_ = L0_.inverse();

			U2[i] = L0_ * U2[i];
			U1[i] = L0_ * U1[i];
			U0[i] = L0_ * U0[i];
			b[i] = L0_ * b[i];
		}

		// Compute the final velocity and poses
		for (int i = numParticle - numStrand + si, li = n - 1; i >= 0; i -= numStrand, --li) {

			i1 = i + numStrand;
			i2 = i1 + numStrand;
			i3 = i2 + numStrand;

			if (li + 1 < n)
				vels[i] -= U0[i] * vels[i1];
			if (li + 2 < n)
				vels[i] -= U1[i] * vels[i2];
			if (li + 3 < n)
				vels[i] -= U2[i] * vels[i3];

			float3 prevPos = prevPoses[i];

			// Rigidness
			poses[i] = lerp(prevPos + vels[i] * t, dTransform * prevPos + dTranslation, rigidness[i]);
		}

		// Apply strain limiting
		if (strainLimitingTolerance > 1.0f) {
			p[0] = poses[si];
			rp[0] = restPoses[si];
			for (int i = si + numStrand; i < numParticle; i += numStrand) {
				p[1] = poses[i];
				rp[1] = restPoses[i];

				float ltol = length(rp[1] - rp[0]) * strainLimitingTolerance;

				float3 d = p[1] - p[0];
				float l = length(d);

				if (l > ltol)
					p[1] = p[0] + d * (ltol / l);

				// Write back
				poses[i] = p[1];

				// Assign for next itereation
				p[0] = p[1];
				rp[0] = rp[1];
			}
		}
	}

	void SelleMassSpringImplicitCudaSolver_resolveStrandDynamics(
			Mat3 **A,
			Mat3 **L,
			Mat3 **U,
			float3 *y,
			float3 *b,
			float3 *poses,
			float3 *prevPoses,
			float3 *restPoses,
			float3 *vels,
			float3 *impulses,
			const float *rigidness,
			Mat3 dTransform,
			float3 dTranslation,
			int numParticle,
			int numStrand,
			int numParticlePerStrand,
			float pmass,
			float damping,
			float kStretch,
			float kBending,
			float kTorsion,
			float strainLimitingTolerance,
			float t,
			int wrapSize
	) {
		// One for each strand
		int numThread = wrapSize * 8;
		int numBlock = (numStrand + numThread - 1) / numThread;

		cudaFuncSetCacheConfig(SelleMassSpringImplicitCudaSolver_resolveStrandDynamicsKernal, cudaFuncCachePreferShared);
		SelleMassSpringImplicitCudaSolver_resolveStrandDynamicsKernal<<<numBlock, numThread, 4 * sizeof(Mat3) * numThread>>>(
				L[0], L[1], L[2], L[3],
				U[0], U[1], U[2],
				poses, prevPoses, restPoses, vels, impulses,
				rigidness, dTransform, dTranslation, numParticle, numStrand, numParticlePerStrand,
				pmass, damping, kStretch, kBending, kTorsion, strainLimitingTolerance, t
		);

		cudaDeviceSynchronize();
	}

	__global__
	void SelleMassSpringImplicitCudaSolver_getVelocityFromPositionKernal(
			const float3 * poses,
			const float3 * prevPoses,
			float3 * vels,
			float tInv,
			int numParticle
	) {
		// One for each particle
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= numParticle)
			return;

		vels[i] = (poses[i] - prevPoses[i]) * tInv;

//		if (i == 0) {
//			for (int k = 0; k < numParticle; ++k) {
//				float3 pos = poses[k];
//				float3 prevPos = prevPoses[k];
//				float3 vel = vels[k];
//				printf("%d: pos: {%f, %f, %f}, prevPos: {%f, %f, %f}, vel: {%f, %f, %f}\n", k, pos.x, pos.y, pos.z, prevPos.x, prevPos.y, prevPos.z, vel.x, vel.y, vel.z);
//			}
//		}
	}

	void SelleMassSpringImplicitCudaSolver_getVelocityFromPosition(
			const float3 * poses,
			const float3 * prevPoses,
			float3 * vels,
			float tInv,
			int numParticle,
			int wrapSize
	) {
		// One for each particle
		int numThread = wrapSize * 32;
		int numBlock = (numParticle + numThread - 1) / numThread;

		SelleMassSpringImplicitCudaSolver_getVelocityFromPositionKernal<<<numBlock, numThread>>>(
				poses, prevPoses, vels, tInv, numParticle
		);
		cudaDeviceSynchronize();
	}
}