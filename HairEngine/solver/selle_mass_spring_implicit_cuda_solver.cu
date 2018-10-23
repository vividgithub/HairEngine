
#include "cudauntime.h"
#include "../util/simple_mat.h"
#include "selleasspringonf.h"
#include "selle_mass_spring_conf.h"

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
			float t
	) {
		// One for each strand
		int si = blockIdx.x * blockDim.x + threadIdx.x;
		if (si >= numStrand)
			break;

		// Alias
		const auto & n = numParticlePerStrand;

		// Initialize the b and A
		float3 p[4]; // Previous poses, store in register
		float3 rp[4]; // Previous rest poses, store in register

		float3 impulse;
		Mat3 dm;
		for (int i = si, li = 0; i < numParticle; i += numStrand, ++li) {

			A[3][i] = Mat3::Diagnoal(pmass + damping * t);
			A[0][i] = A[1][i] = A[2][i] = A[4][i] = A[5][i] = A[6][i] = Mat3::Zero();

			b[i] = pmass * vels[i] + impulses[i] * t;

			// Apply spring force, since b[pi] and A[xxx][pi] has been correctly initialized, it is safe to modify
			p[3] = poses[i];
			rp[3] = restPoses[i];

			// Stretch spring
			if (li >= 1) {
				getSpringInfo(p[2], p[3], rp[2], rp[3], t, kStretch, impulse, dm);

				auto pi = i - numStrand;
				b[pi] += impulse;
				b[i] -= impulse;

				A[3][pi] += dm;
				A[3][i] += dm;
				A[4][pi] -= dm;
				A[2][i] -= dm;
			}

			// Bending spring
			if (li >= 2) {
				getSpringInfo(p[1], p[3], rp[1], rp[3], t, kBending, impulse, dm);

				auto pi = i - 2 * numStrand;
				b[pi] += impulse;
				b[i] -= impulse;

				A[3][pi] += dm;
				A[3][i] += dm;
				A[5][pi] -= dm;
				A[1][i] -= dm;
			}

			// Torsion spring
			if (li >= 3) {
				getSpringInfo(p[0], p[3], rp[0], rp[3], t, kTorsion, impulse, dm);

				auto pi = i - 3 * numStrand;
				b[pi] += impulse;
				b[i] -= impulse;

				A[3][pi] += dm;
				A[3][i] += dm;
				A[6][pi] -= dm;
				A[0][i] -= dm;
			}

			p[0] = p[1]; p[1] = p[2]; p[2] = p[3];
			rp[0] = rp[1]; rp[1] = rp[2]; rp[2] = rp[3];
		}

		// Initialize b and A of the strand root
		b[si] = dTransform * prevPoses[si] + dTranslation;
		A[0][si] = A[1][si] = A[2][si] = A[4][si] = A[5][si] = Mat3::Zero();
		A[3][si] = Mat3::Identity();

		// Heptadignoal solver
		for (int i = si, li = 0; i < numParticle; i += numStrand, ++li) {
			//compute L3
			L[3][i] = li >= 3 ? A[0][i] : Mat3::Zero();

			//compute L2
			L[2][i] = li >= 2 ? A[1][i] : Mat3::Zero();
			if (li >= 3)
				L[2][i] -= L[3][i] * U[0][i - 3];

			//compute L1
			L[1][i] = li >= 1 ? A[2][i] : Mat3::Zero();
			if (li >= 2)
				L[1][i] -= L[2][i] * U[0][i - 2];
			if (li >= 3)
				L[1][i] -= L[3][i] * U[1][i - 3];

			//compute L0
			L[0][i] = A[3][i];
			if (li >= 1)
				L[0][i] -= L[1][i] * U[0][i - 1];
			if (li >= 2)
				L[0][i] -= L[2][i] * U[1][i - 2];
			if (li >= 3)
				L[0][i] -= L[3][i] * U[2][i - 3];

			//compute L0i
			L[4][i] = L[0][i].inverse();

			//compute U2
			U[2][i] = li + 3 < n ? L[4][i] * A[6][i] : Mat3::Zero();

			//compute U1
			U[1][i] = li + 2 < n ? A[5][i] : Mat3::Zero();
			if (li >= 1)
				U[1][i] -= L[1][i] * U[2][i - 1];
			U[1][i] = L[4][i] * U[1][i];

			//compute U0
			U[0][i] = li + 1 < n ? A[4][i] : Mat3::Zero();
			if (li >= 1)
				U[0][i] -= L[1][i] * U[1][i - 1];
			if (li >= 2)
				U[0][i] -= L[2][i] * U[2][i - 2];
			U[0][i] = L[4][i] * U[0][i];

			//compute y
			y[i] = b[i];
			if (li >= 1)
				y[i] -= L[1][i] * y[i - 1];
			if (li >= 2)
				y[i] -= L[2][i] * y[i - 2];
			if (li >= 3)
				y[i] -= L[3][i] * y[i - 3];
			y[i] = L[4][i] * y[i];
		}

		// Compute the final velocity and poses
		for (int i = numParticle - numStrand + si, li = n - 1; i >= 0; i -= numStrand, --li) {
			vels[i] = y[i];
			if (li + 1 < n)
				vels[i] -= U[0][i] * x[i + 1];
			if (li + 2 < n)
				vels[i] -= U[1][i] * x[i + 2];
			if (li + 3 < n)
				vels[i] -= U[2][i] * x[i + 3];

			float3 prevPos = prevPoses[i];

			// Rigidness
			poses[i] = lerp(prevPos + vels[i] * t, dTransform * prevPos + dTranslation, rigidness[i]);
		}

		// Apply strain limiting
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
				A, L, U, y, b, poses, prevPoses, restPoses, vels, impulses,
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