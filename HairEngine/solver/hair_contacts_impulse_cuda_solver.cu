#ifdef HAIRENGINE_ENABLE_CUDA

#include <cstdio>
#include "../util/helper_math.h"

namespace HairEngine {

	__global__
	void HairContactsImpulseCudaSolver_updateContactsSpringKernal(const int *hashSegStarts, const int *hashSegEnds,
	                                                        const int *sids, const float3 *midpoints,
	                                                        const int *segStrandIndices, int *contacts,
	                                                        int *numContacts, float3 *parImpulses,
	                                                        int numSegment, float lCreate, float lBreak,
	                                                        int maxContacts, float k, float3 dInv,
	                                                        int hashShift) {
		// One for each segments
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= numSegment)
			return;
		int sid1 = sids[idx];
		int sid1StrandIndex = segStrandIndices[sid1];

		float3 center = midpoints[sid1];

		// Delete the old segments
		int *contactsStarts = contacts + sid1 * maxContacts;
		int n = 0; // An updated numContact

		// TODO: Move it to the back
		float3 force = make_float3(0.0f);
		for (int i = 0; i < numContacts[sid1]; ++i) {
			int sid2 = contactsStarts[i];

			float3 d = midpoints[sid2] - center;
			float l = length(d) + 1e-30f;
			d /= l;

			if (l < lBreak) {
				force += d * (l - lCreate);
				contactsStarts[n++] = contactsStarts[i];
			}
		}

		force *= k;

		// Add the force to the impulse array with pid = sid +
		int pid = sid1 + segStrandIndices[sid1];
		float *impulseData = reinterpret_cast<float*>(parImpulses + pid);

		// The first connected particle
		atomicAdd(impulseData, force.x);
		atomicAdd(impulseData + 1, force.y);
		atomicAdd(impulseData + 2, force.z);

		// The second connected particle
		atomicAdd(impulseData + 3, force.x);
		atomicAdd(impulseData + 4, force.y);
		atomicAdd(impulseData + 5, force.z);

		// Search for new contacts
		int3 index3Max, index3Min;
		int n_;

		if (n == maxContacts)
			goto WriteBackNumContacts;

		index3Max = make_int3((center + lCreate) * dInv); // Ceiling
		index3Min = make_int3((center - lCreate) * dInv); // Floor

		n_ = n; // Store current n to n_, so the check iteration will only need to iterate to n_

		for (int ix = index3Min.x; ix <= index3Max.x; ++ix)
			for (int iy = index3Min.y; iy <= index3Max.y; ++iy)
				for (int iz = index3Min.z; iz <= index3Max.z; ++iz) {
					// Get the hash value
					uint32_t hash = getHashValueFromIndex3(make_int3(ix, iy, iz), hashShift);
					int hashStart = hashSegStarts[hash];
					if (hashStart == 0xffffffff)
						continue;
					int hashEnd = hashSegEnds[hash];

					// Check from start to end
					for (int hashIt = hashStart; hashIt != hashEnd; ++hashIt) {
						int sid2 = sids[hashIt];

						if (sid2 == sid1 || segStrandIndices[sid2] == sid1StrandIndex)
							continue;

						if (length(center - midpoints[sid2]) < lCreate) {
							// Check whether it is added
							bool added = false;
							for (int i = 0; i < n_; ++i) if (contactsStarts[i] == sid2) {
									added = true;
									break;
							}

							if (!added) {
								contactsStarts[n++] = sid2;
								if (n == maxContacts)
									goto WriteBackNumContacts;
							}
						}
					}
				}

		WriteBackNumContacts:

		numContacts[sid1] = n;
	}

	void HairContactsImpulseCudaSolver_updateContactsSpring(const int *hashSegStarts, const int *hashSegEnds,
	                                                        const int *sids, const float3 *midpoints,
	                                                        const int *segStrandIndices, int *contacts,
	                                                        int *numContacts, float3 *parImpulses,
	                                                        int numSegment, float lCreate, float lBreak,
	                                                        int maxContacts, float k, float3 dInv,
	                                                        int hashShift, int wrapSize) {
		int numThread = wrapSize * 32;
		int numBlock = (numSegment + numThread - 1) / numThread;

		HairContactsImpulseCudaSolver_updateContactsSpringKernal<<<numBlock, numThread>>>(hashSegStarts, hashSegEnds,
				sids, midpoints, segStrandIndices, contacts, numContacts, parImpulses, numSegment, lCreate,
				lBreak, maxContacts, k, dInv, hashShift);
		cudaDeviceSynchronize();
	}
}

#endif