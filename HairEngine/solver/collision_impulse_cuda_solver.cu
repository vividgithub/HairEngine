
#ifdef HAIRENGINE_ENABLE_CUDA

#include <cstdio>
#include <string>
#include "../util/cuda_helper_math.h"

namespace HairEngine {

	constexpr const int CUDA_HAIR_COLLISIONS_IMPULSE_SOLVER_MAX_COLLISIONS_LIMITATION = 20;

	__device__ __forceinline__
	float3 D(float3 s1p1, float3 s1p2, float3 s2p1, float3 s2p2, float2 t) {
		return lerp(s2p1, s2p2, t.y) - lerp(s1p1, s1p2, t.x);
	}

	__device__ __forceinline__
	float2 CPA(float3 p0, float3 p1, float3 q0, float3 q1) {
		auto p0p1 = p1 - p0;
		auto q0q1 = q1 - q0;
		auto q0p0 = p0 - q0;

		float a = dot(p0p1, p0p1);
		float b = dot(p0p1, q0q1);
		float c = dot(q0q1, q0q1);
		float d = dot(p0p1, q0p0);
		float e = dot(q0q1, q0p0);

		float f1 = b * b - a * c;

		float2 t;
		t.x = (c * d - b * e) / f1;
		t.y = (b * d - a * e) / f1;

		return t;
	}

	__global__
	void CollisionImpulseCudaSolver_updateCollisionSpringKernal(
			const int *hashSegStarts,
			const int *hashSegEnds,
			const int *sids,
			const float3 *parPoses,
			const float3 *parVels,
			const int *segStrandIndices,
			int *collisions,
			float2 *cpas,
			float3 *dns,
			float *l0s,
			int *numCollisions,
			float3 *segForces,
			int *numSegForces,
			int numSegment,
			int numStrand,
			int maxCollisions,
			float3 dInv,
			int hashShift,
			float time
	) {
		// One for each segment
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= numSegment)
			return;

		int sid1 = sids[idx];
		int segCollisions[CUDA_HAIR_COLLISIONS_IMPULSE_SOLVER_MAX_COLLISIONS_LIMITATION];
		int n_ = numCollisions[sid1];
		int n = 0;

		float3 s1p1 = parPoses[sid1];
		float3 s1p2 = parPoses[sid1 + numStrand];
		float3 s1v1 = parVels[sid1];
		float3 s1v2 = parVels[sid1 + numStrand];

		// Load the current collisions, check whether the spring should be broken, it not so,
		// Load it into the segCollisions registers and add the force to the segForce and
		// numSegForces
		int go = sid1 * maxCollisions; // Global offset, alias
		for (int i = 0; i < n_; ++i) {
			int off = go + i; // Real offset, alias

			int sid2 = collisions[off]; // Fetch sid
			float2 t = cpas[off]; // Fetch cpas
			float3 dn = dns[off]; // Fetch dn

			float3 s2p1 = parPoses[sid2];
			float3 s2p2 = parPoses[sid2 + numStrand];

			// Compute d (current direction) and d_est(the estimated direction)
			float3 d = D(s1p1, s1p2, s2p1, s2p2, t);
			float3 d_est = d + time * D(s1v1, s1v2, parVels[sid2], parVels[sid2 + numStrand], t);

			if (dot(d_est, dn) < 0.0f) {
				int writeOffset = go + n;
				float l0 = l0s[off]; // Fetch l0s

				// Copy to the new position
				collisions[writeOffset] = sid2;
				cpas[writeOffset] = t;
				dns[writeOffset] = dn;
				l0s[writeOffset] = l0;

				segCollisions[n] = sid2; // Load the index in the register

				++n;

				// Compute the force
				float l = length(d) + 1e-30f;
				float3 force = d * (1.0f - l0 / l); // Not multiply k

				// Add the force to the segment force
				atomicAdd(numSegForces + sid1, 1);
				atomicAdd(numSegForces + sid2, 1);

				auto segForce1 = reinterpret_cast<float *>(segForces + sid1);
				auto segForce2 = reinterpret_cast<float *>(segForces + sid2);

				atomicAdd(segForce1, force.x);
				atomicAdd(segForce1 + 1, force.y);
				atomicAdd(segForce1 + 2, force.z);
				force = -force;
				atomicAdd(segForce2, force.x);
				atomicAdd(segForce2 + 1, force.y);
				atomicAdd(segForce2 + 2, force.z);
			}
		}

		// Full, just return
		if (n == maxCollisions)
			return;

		n_ = n; // Store the current n to n_, and check iteration will only iterate to n_

		// We don't need to load the midpoints since we can compute it by s1p1 and s1p2
		float3 center = (s1p1 + s1p2) / 2.0f;
		float radius = length(s1p2 - s1p1) / 2.0f;

		int3 index3Center = make_int3(center * dInv);
		int3 index3Max = make_int3((center + radius) * dInv); // Ceiling
		int3 index3Min = make_int3((center - radius) * dInv); // Floor
		int3 offset = max(index3Max - index3Center, index3Center - index3Min);

		// The long expression here is trying to iterate ix from indexCenter.x --> indexCenter.x + 1 --->
		// indexCenter.x - 1 --> ... until all the cells have been iterated
		int ix, iy, iz, ox, oy, oz;
		for (ox = 0, ix = index3Center.x; abs(ox) <= offset.x; ox = ox > 0 ? -ox : -ox + 1, ix = index3Center.x + ox) if (ix >= index3Min.x && ix <= index3Max.x)
			for (oy = 0, iy = index3Center.y; abs(oy) <= offset.y; oy = oy > 0 ? -oy : -oy + 1, iy = index3Center.y + oy) if (iy >= index3Min.y && iy <= index3Max.y)
				for (oz = 0, iz = index3Center.z; abs(oz) <= offset.z; oz = oz > 0 ? -oz : -oz + 1, iz = index3Center.z + oz) if (iz >= index3Min.z && iz <= index3Max.z) {
					// Get the hash value
					uint32_t hash = getHashValueFromIndex3(make_int3(ix, iy, iz), hashShift);
					int hashStart = hashSegStarts[hash];
					if (hashStart == 0xffffffff)
						continue;
					int hashEnd = hashSegEnds[hash];

					// Check from start to end
					for (int hashIt = hashStart; hashIt != hashEnd; ++hashIt) {
						int sid2 = sids[hashIt];

						// We only add sid2 > sid1 segment
						if (sid2 <= sid1 || segStrandIndices[sid2] == segStrandIndices[sid1])
							continue;

						float3 s2p1 = parPoses[sid2];
						float3 s2v1 = parVels[sid2];
						float3 s2p2 = parPoses[sid2 + numStrand];
						float3 s2v2 = parVels[sid2 + numStrand];

						float2 t = CPA(s1p1, s1p2, s2p1, s2p2);

						if (!(t.x >= 0.0f && t.x <= 1.0f && t.y >= 0.0f && t.y <= 1.0f))
							continue;

						float3 dn = D(s1p1, s1p2, s2p1, s2p2, t);
						float l0 = length(dn);
						dn /= l0;

						// Check

						// Get the projection coefficient, and make sure it is >= 0.0f
						float co_dnv = dot(lerp(s2v1, s2v2, t.y), dn) - dot(lerp(s1v1, s1v2, t.x), dn);
						if (co_dnv >= 0.0f || -co_dnv * time < l0)
							continue;

						// Check not added
						bool added = false;
						for (int i = 0; i < n_; ++i) if (segCollisions[i] == sid2) {
								added = true;
								break;
							}
						if (added)
							continue;

						// Add to the collision
						int writeOffset = go + n;
						collisions[writeOffset] = sid2;
						cpas[writeOffset] = t;
						dns[writeOffset] = dn;
						l0s[writeOffset] = l0;
						++n;

						if (n == maxCollisions)
							goto WriteBack;
					}
				}

		WriteBack:
			numCollisions[sid1] = n;
	}

	__device__ __forceinline__
	float3 getForce(float3 totalForce, int forceCount, int maxForceCount) {
		return (forceCount == 0) ? make_float3(0.0f) : fminf(static_cast<float>(maxForceCount) / forceCount, 1.0f) * totalForce;
	}

	__global__
	void CollisionImpulseCudaSolver_applyCollisionForceToParticlesKernal(
			const float3 *segForces,
			int *numSegForces,
			float3 *parImpulses,
			float k,
			int maxForceCount,
			int numSegment,
			int numStrand
	) {
		// One for each particles, so that no atomicAdd is needed
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= numSegment + numStrand)
			return;

		float3 force = make_float3(0.0f);
		int forceCount = 0;

		// Not the last segment in the strand
		if (i < numSegment) {
			force += segForces[i];
			forceCount += numSegForces[i];
		}

		// Not the first segment in the strand
		int i1 = i - numStrand;
		if (i1 >= 0) {
			force += segForces[i1];
			forceCount += numSegForces[i1];
		}

		parImpulses[i] += k * getForce(force, forceCount, maxForceCount);
	}

	void CollisionImpulseCudaSolver_updateCollisionSpring(
			const int *hashSegStarts,
			const int *hashSegEnds,
			const int *sids,
			const float3 *parPoses,
			const float3 *parVels,
			float3 *parImpulses,
			const int *segStrandIndices,
			int *collisions,
			float2 *cpas,
			float3 *dns,
			float *l0s,
			int *numCollisions,
			float3 *segForces,
			int *numSegForces,
			int numSegment,
			int numStrand,
			int maxCollisions,
			int maxForceCount,
			float k,
			float3 dInv,
			int hashShift,
			float time,
			int wrapSize
	) {
		int numThread1 = 32 * wrapSize;
		int numBlock1 = (numSegment + numThread1 - 1) / numThread1;

		// Set the force and number of forces to all zero
		cudaMemset(segForces, 0x00, sizeof(float3) * numSegment);
		cudaMemset(numSegForces, 0x00, sizeof(int) * numSegment);

		CollisionImpulseCudaSolver_updateCollisionSpringKernal<<<numBlock1, numThread1>>>(
				hashSegStarts,
				hashSegEnds,
				sids,
				parPoses,
				parVels,
				segStrandIndices,
				collisions,
				cpas,
				dns,
				l0s,
				numCollisions,
				segForces,
				numSegForces,
				numSegment,
				numStrand,
				maxCollisions,
				dInv,
				hashShift,
				time
		);

		cudaDeviceSynchronize();

		int numThread2 = 32 * wrapSize;
		int numBlock2 = (numSegment + numStrand + numThread2 - 1) / numThread2;

		CollisionImpulseCudaSolver_applyCollisionForceToParticlesKernal<<<numBlock2, numThread2>>>(
				segForces,
				numSegForces,
				parImpulses,
				k,
				maxForceCount,
				numSegment,
				numStrand
		);

		cudaDeviceSynchronize();
	}
}

#endif