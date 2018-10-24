
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
			Mat3 *A0, Mat3 *A1, Mat3 *A2, Mat3 *A3, Mat3 *A4, Mat3 *A5, Mat3 *A6,
			Mat3 *L0, Mat3 *L1, Mat3 *L2, Mat3 *L3, Mat3 *L4,
			Mat3 *U0, Mat3 *U1, Mat3 *U2,
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
			float t
	) {
		// One for each strand
		int si = blockIdx.x * blockDim.x + threadIdx.x;
		if (si >= numStrand)
			return;

		// Alias
		const auto & n = numParticlePerStrand;

		// Initialize the b and A
		float3 p[4]; // Previous poses, store in register
		float3 rp[4]; // Previous rest poses, store in register

		float3 impulse;
		Mat3 dm;
		for (int i = si, li = 0; i < numParticle; i += numStrand, ++li) {

			A3[i] = Mat3::Diagnoal(pmass + damping * t);
			A0[i] = A1[i] = A2[i] = A4[i] = A5[i] = A6[i] = Mat3::Zero();

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

				A3[pi] += dm;
				A3[i] += dm;
				A4[pi] -= dm;
				A2[i] -= dm;
			}

//			if (si == 0) {
//				printf("p[2]: {%f, %f, %f}, p[3]: {%f, %f, %f}, rp[2]: {%f, %f, %f}, rp[3]: {%f, %f, %f}\n", p[2].x, p[2].y, p[2].z, p[3].x, p[3].y, p[3].z, rp[2].x, rp[2].y, rp[2].z, rp[3].x, rp[3].y, rp[3].z);
//				printf("impulse: {%f, %f, %f}\n", impulse.x, impulse.y, impulse.z);
//				printf("dm: "); dm.print(); printf("\n");
//			}
//
			// Bending spring
			if (li >= 2) {
				getSpringInfo(p[1], p[3], rp[1], rp[3], t, kBending, impulse, dm);

				auto pi = i - 2 * numStrand;
				b[pi] += impulse;
				b[i] -= impulse;

				A3[pi] += dm;
				A3[i] += dm;
				A5[pi] -= dm;
				A1[i] -= dm;
			}

			// Torsion spring
			if (li >= 3) {
				getSpringInfo(p[0], p[3], rp[0], rp[3], t, kTorsion, impulse, dm);

				auto pi = i - 3 * numStrand;
				b[pi] += impulse;
				b[i] -= impulse;

				A3[pi] += dm;
				A3[i] += dm;
				A6[pi] -= dm;
				A0[i] -= dm;
			}

			p[0] = p[1]; p[1] = p[2]; p[2] = p[3];
			rp[0] = rp[1]; rp[1] = rp[2]; rp[2] = rp[3];
		}

		// Initialize b and A of the strand root
		b[si] = (dTransform * prevPoses[si] + dTranslation - prevPoses[si]) / t;
		A0[si] = A1[si] = A2[si] = A4[si] = A5[si] = A6[si] = Mat3::Zero();
		A3[si] = Mat3::Identity();

		// Heptadignoal solver
		int i1, i2, i3;

		for (int i = si, li = 0; i < numParticle; i += numStrand, ++li) {

			i1 = i - numStrand;
			i2 = i1 - numStrand;
			i3 = i2 - numStrand;

			//compute L3
			L3[i] = li >= 3 ? A0[i] : Mat3::Zero();

			//compute L2
			L2[i] = li >= 2 ? A1[i] : Mat3::Zero();
			if (li >= 3)
				L2[i] -= L3[i] * U0[i3];

			//compute L1
			L1[i] = li >= 1 ? A2[i] : Mat3::Zero();
			if (li >= 2)
				L1[i] -= L2[i] * U0[i2];
			if (li >= 3)
				L1[i] -= L3[i] * U1[i3];

			//compute L0
			L0[i] = A3[i];
			if (li >= 1)
				L0[i] -= L1[i] * U0[i1];
			if (li >= 2)
				L0[i] -= L2[i] * U1[i2];
			if (li >= 3)
				L0[i] -= L3[i] * U2[i3];

			//compute L0i
			L4[i] = L0[i].inverse();

			//compute U2
			U2[i] = li + 3 < n ? L4[i] * A6[i] : Mat3::Zero();

			//compute U1
			U1[i] = li + 2 < n ? A5[i] : Mat3::Zero();
			if (li >= 1)
				U1[i] -= L1[i] * U2[i1];
			U1[i] = L4[i] * U1[i];

			//compute U0
			U0[i] = li + 1 < n ? A4[i] : Mat3::Zero();
			if (li >= 1)
				U0[i] -= L1[i] * U1[i1];
			if (li >= 2)
				U0[i] -= L2[i] * U2[i2];
			U0[i] = L4[i] * U0[i];

			//compute y
			y[i] = b[i];
			if (li >= 1)
				y[i] -= L1[i] * y[i1];
			if (li >= 2)
				y[i] -= L2[i] * y[i2];
			if (li >= 3)
				y[i] -= L3[i] * y[i3];
			y[i] = L4[i] * y[i];
		}

		// Compute the final velocity and poses
		for (int i = numParticle - numStrand + si, li = n - 1; i >= 0; i -= numStrand, --li) {

			i1 = i + numStrand;
			i2 = i1 + numStrand;
			i3 = i2 + numStrand;

			vels[i] = y[i];
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

//		if (si == 0) {
//			printf("Initialize A and b:\n");
//			for (int i = si, li = 0; i < numParticle; i += numStrand, ++li) {
//
//				printf("%d(%d)\n", i, li);
//
//				for (int k = 0; k < 7; ++k) {
//					printf("\t\tA[%d][%d]: ", k, i);
//					A[k][i].print();
//					printf("\n");
//				}
//
//				for (int k = 0; k < 5; ++k) {
//					printf("\t\tL[%d][%d]: ", k, i);
//					L[k][i].print();
//					printf("\n");
//				}
//
//				for (int k = 0; k < 3; ++k) {
//					printf("\t\tU[%d][%d]: ", k, i);
//					U[k][i].print();
//					printf("\n");
//				}
//
//				printf("\t\tb[%d]: {%f, %f, %f}\n", i, b[i].x, b[i].y, b[i].z);
//				printf("\t\ty[%d]: {%f, %f, %f}\n", i, y[i].x, y[i].y, y[i].z);
//				printf("\t\tx[%d]: {%f, %f, %f}\n", i, vels[i].x, vels[i].y, vels[i].z);
//			}
//		}

//		if (si == 0) {
//			for (int i = si; i < numParticle; i += numStrand) {
//				float3 prevPos = prevPoses[i];
//				float3 restPos = restPoses[i];
//				float3 pos = poses[i];
//				float3 vel = vels[i];
//				float3 impulse =impulses[i];
//
//				printf("Particle(%d) {prevPos: {%f, %f, %f}, restPos: {%f, %f, %f}, pos: {%f, %f, %f}, vel: {%f, %f, %f}, impulse: {%f, %f, %f}, rigidness: %f}\n", i, prevPos.x, prevPos.y, prevPos.z, restPos.x, restPos.y, restPos.z, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, impulse.x, impulse.y, impulse.z, rigidness[i]);
//			}
//
//			printf("dTransform: "); dTransform.print(); printf("\n");
//			printf("dTranslation: {%f, %f, %f}\n", dTranslation.x, dTranslation.y, dTranslation.z);
//		}
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

		SelleMassSpringImplicitCudaSolver_resolveStrandDynamicsKernal<<<numBlock, numThread>>>(
				A[0], A[1], A[2], A[3], A[4], A[5], A[6],
				L[0], L[1], L[2], L[3], L[4],
				U[0], U[1], U[2],
				y, b, poses, prevPoses, restPoses, vels, impulses,
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