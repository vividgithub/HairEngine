#ifdef HAIRENGINE_ENABLE_CUDA

#include <cstdio>
#include <stdexcept>
#include "../util/cuda_helper_math.h"

namespace HairEngine {

	constexpr const int CUDA_HAIR_CONTACTS_IMPULSE_SOLVER_MAX_CONTACTS_LIMITATION = 15;

	__global__
	void HairContactsImpulseCudaSolver_updateContactsSpringKernal(const int *hashSegStarts, const int *hashSegEnds,
	                                                        const int *sids, const float3 *midpoints,
	                                                        const int *segStrandIndices, int *contacts,
	                                                        int *numContacts, float3 *parImpulses,
	                                                        int numSegment, int numStrand, float lCreate, float lBreak,
	                                                        int maxContacts, float k, float3 dInv,
	                                                        int hashShift) {

		// One for each segments
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= numSegment)
			return;

		int sid1 = sids[idx];
		int sid1StrandIndex = segStrandIndices[sid1];

		float3 center = midpoints[sid1];
		float3 force = make_float3(0.0f);

		// Allocate local memory for the contact ids
		int segContacts[CUDA_HAIR_CONTACTS_IMPULSE_SOLVER_MAX_CONTACTS_LIMITATION];

		int n = numContacts[sid1];
		int *segContactsGlobal = contacts + sid1 * maxContacts;
		for (int i = 0; i < n; ++i)
			segContacts[i] = segContactsGlobal[i];

		// Delete the old segments
		n = 0; // An updated numContact

		for (int i = 0; i < numContacts[sid1]; ++i) {
			int sid2 = segContacts[i];

			float3 d = midpoints[sid2] - center;
			float l = length(d) + 1e-30f;

			if (l < lBreak) {
				segContacts[n++] = segContacts[i];
				force += d * (1.0f - lCreate / l);
			}
		}

		// Search for new contacts
		int3 index3Max, index3Min, index3Center;
		int n_;
		int3 offset;

		if (n == maxContacts)
			goto ComputeForce;


		index3Center = make_int3(center * dInv);
		index3Max = make_int3((center + lCreate) * dInv); // Ceiling
		index3Min = make_int3((center - lCreate) * dInv); // Floor
		offset = max(index3Max - index3Center, index3Center - index3Min);

		n_ = n; // Store current n to n_, so the check iteration will only need to iterate to n_

		int ix, iy, iz, ox, oy, oz;
		// The long expression here is trying to iterate ix from indexCenter.x --> indexCenter.x + 1 --->
		// indexCenter.x - 1 --> ... until all the cells have been iterated
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
						int sid2StrandIndex = segStrandIndices[sid2];

						if (sid2 == sid1 || sid2StrandIndex == sid1StrandIndex)
							continue;

						float3 d = midpoints[sid2] - center;
						float l = length(d) + 1e-30f;

						if (l < lCreate) {
							// Check whether it is added
							bool added = false;
							for (int i = 0; i < n_; ++i) if (segStrandIndices[segContacts[i]] == sid2StrandIndex) {
									added = true;
									break;
							}

							if (!added) {
								segContacts[n++] = sid2;
								force += d * (1.0f - lCreate / l);
								if (n == maxContacts)
									goto ComputeForce;
							} // End !added
						} // End l < lCreate
					} // End hash iteration
				} // End ox, oy, oz

		ComputeForce:

		force *= k;

		// Write the force to the segment and to the particle
		float *par1Impulse = reinterpret_cast<float*>(parImpulses + sid1);
		float *par2Impulse = reinterpret_cast<float*>(parImpulses + sid1 + numStrand);

		atomicAdd(par1Impulse, force.x);
		atomicAdd(par1Impulse + 1, force.y);
		atomicAdd(par1Impulse + 2, force.z);

		atomicAdd(par2Impulse, force.x);
		atomicAdd(par2Impulse + 1, force.y);
		atomicAdd(par2Impulse + 2, force.z);

		// Write back
		numContacts[sid1] = n;
		for (int i = 0; i < n; ++i)
			segContactsGlobal[i] = segContacts[i];
	}

	void HairContactsImpulseCudaSolver_updateContactsSpring(const int *hashSegStarts, const int *hashSegEnds,
	                                                        const int *sids, const float3 *midpoints,
	                                                        const int *segStrandIndices, int *contacts,
	                                                        int *numContacts, float3 *parImpulses,
	                                                        int numSegment, int numStrand, float lCreate, float lBreak,
	                                                        int maxContacts, float k, float3 dInv,
	                                                        int hashShift, int wrapSize) {

		if (maxContacts > CUDA_HAIR_CONTACTS_IMPULSE_SOLVER_MAX_CONTACTS_LIMITATION)
			throw std::runtime_error("[HairContactsImpulseSolver:updateContactsSpring] Max hair contacts large than limitation");

		int numThread = wrapSize * 32;
		int numBlock = (numSegment + numThread - 1) / numThread;

		HairContactsImpulseCudaSolver_updateContactsSpringKernal<<<numBlock, numThread>>>(hashSegStarts, hashSegEnds,
				sids, midpoints, segStrandIndices, contacts, numContacts, parImpulses, numSegment, numStrand, lCreate,
				lBreak, maxContacts, k, dInv, hashShift);
		cudaDeviceSynchronize();
	}
}

#endif