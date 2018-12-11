//
// Created by vivi on 2018/10/7.
//

#pragma once

#ifdef HAIRENGINE_ENABLE_CUDA

#include <VPly/vply.h>

#include <chrono>
#include "../util/cudautil.h"
#include "solver.h"
#include "visualizer.h"
#include "integrator.h"
#include "cuda_based_solver.h"
#include "cuda_segment_midpoint_computer.h"

#include "../accel/particle_spatial_hashing.h"

namespace HairEngine {

	void HairContactsImpulseCudaSolver_updateContactsSpring(const int *hashSegStarts, const int *hashSegEnds,
	                                                        const int *sids, const float3 *midpoints,
	                                                        const int *segStrandIndices, int *contacts,
	                                                        int *numContacts, float3 *parImpulses,
	                                                        int numSegment, int numStrand, float lCreate, float lBreak,
	                                                        int maxContacts, float k, float3 dInv,
	                                                        int hashShift, int wrapSize);

	class HairContactsImpulseCudaSolver: public CudaBasedSolver {
	HairEngine_Public:
		HairContactsImpulseCudaSolver(CudaSegmentMidpointComputer *smc, float creatingDistance,
				float breakingDistance, int maxContactPerSegment, float kContactSpring, float resolution = 1.0, int wrapSize = 8):
				CudaBasedSolver(Pos_ | Impulse_),
				smc(smc),
				lCreate(creatingDistance),
				lBreak(breakingDistance),
				maxContacts(maxContactPerSegment),
				k(kContactSpring),
				resolution(resolution),
				wrapSize(wrapSize) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {
			CudaBasedSolver::setup(hair, currentTransform);

			// Allocate the space for the particles
			contacts = CudaUtility::allocateCudaMemory<int>(hair.nsegment * maxContacts);
			numContacts = CudaUtility::allocateCudaMemory<int>(hair.nsegment);

			// Set the current contacts number to zero
			cudaMemset(numContacts, 0, sizeof(int) * hair.nsegment);

			// Build particle spatial hashing
			float3 d = make_float3(lCreate / resolution);
			psh = new ParticleSpatialHashing(hair.nsegment, d);
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			// Update the particle spatial hashing
			psh->update(smc->midpoints);

			auto startTime = std::chrono::high_resolution_clock::now();

			// Update the contacts springs
			HairContactsImpulseCudaSolver_updateContactsSpring(psh->hashParStartsDevice, psh->hashParEndsDevice,
					psh->pidsDevice, smc->midpoints, smc->segStrandIndices, contacts, numContacts, cmc->parImpulses,
					hair.nsegment, cmc->numStrand, lCreate, lBreak, maxContacts, k, psh->dInv, psh->numHashShift, wrapSize);

			auto endTime = std::chrono::high_resolution_clock::now();

			auto diffInUs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
			printf("[HairContactsImpulseCudaSolver] Timing: %lld ms(%lld us)\n", diffInUs.count() / 1000, diffInUs.count());
		}

		void tearDown() override {
			cudaFree(contacts);
			cudaFree(numContacts);

			delete psh;
		}

	//HairEngine_Protected:
		CudaSegmentMidpointComputer *smc;

		ParticleSpatialHashing *psh = nullptr;

		float lCreate; ///< The creating distance
		float lBreak; ///< The breaking distance
		int maxContacts; ///< Max contacts per segment
		float k; ///< The stiffness of the contact spring
		float resolution; ///< The query resolution for the particle spatial hashing

		int wrapSize; ///< The cuda computing thread for each thread block

		/// The contact spring for segment i, size equals to "nsegment * maxContacts",
		/// so for segment i, its connected particle global index are stored in
		/// [i * maxContacts, (i + 1) * maxContacts). We only need to store the global index. Allocate in the GPU
		int *contacts = nullptr;

		/// The number of contacts for the segment
		int *numContacts = nullptr;
	};

	class HairContactsImpulseCudaVisualizer: public Visualizer {
	HairEngine_Public:

		HairContactsImpulseCudaVisualizer(const std::string &directory, const std::string &filenameTemplate, float timestep,
		                                  HairContactsImpulseCudaSolver *impulseCudaSolver) :
		                                  Visualizer(directory, filenameTemplate, timestep),
		                                  impulseCudaSolver(impulseCudaSolver) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {

			if (!impulseCudaSolver)
				return;

			// Allocate the space for contacts and numContacts
			contacts = new int[hair.nsegment * impulseCudaSolver->maxContacts];
			numContacts = new int[hair.nsegment];
			midpoints = new float3[hair.nsegment];
		}

		void tearDown() override {
			delete[] contacts;
			delete[] numContacts;
			delete[] midpoints;
		}

		void visualize(std::ostream &os, Hair &hair, const IntegrationInfo &info) override {

			if (!impulseCudaSolver)
				return;


			// Copy the contacts and numContacts
			CudaUtility::copyFromDeviceToHost(contacts, impulseCudaSolver->contacts, hair.nsegment * impulseCudaSolver->maxContacts);
			CudaUtility::copyFromDeviceToHost(numContacts, impulseCudaSolver->numContacts, hair.nsegment);
			CudaUtility::copyFromDeviceToHost(midpoints, impulseCudaSolver->smc->midpoints, hair.nsegment);

			// Visualize
			int totalContacts = 0;
			for (int sid1 = 0; sid1 < hair.nsegment; ++sid1) {

				auto midpoint1 = midpoints[sid1];

				totalContacts += numContacts[sid1];

				for (int t = 0; t < numContacts[sid1]; ++t) {
					int sid2 = contacts[sid1 * impulseCudaSolver->maxContacts + t];
					auto midpoint2 = midpoints[sid2];

					VPly::writeLine(
							os,
							VPly::VPlyVector3f(midpoint1.x, midpoint1.y, midpoint1.z),
							VPly::VPlyVector3f(midpoint2.x, midpoint2.y, midpoint2.z)
//							VPly::VPlyFloatAttr("l0", impulseCudaSolver->lCreate),
//							VPly::VPlyFloatAttr("l", length(midpoint1 - midpoint2)),
//							VPly::VPlyIntAttr("fromid", sid1),
//							VPly::VPlyIntAttr("toid", sid2),
//							VPly::VPlyIntAttr("type", 10)
 					);
				}
			}

			printf("[HairContactsImpulseCudaVisualizer] Total contacts: %d, average contacts: %f\n",
					totalContacts, static_cast<float>(totalContacts) / hair.nsegment);
		}

	HairEngine_Protected:
		HairContactsImpulseCudaSolver *impulseCudaSolver = nullptr;
		int *contacts = nullptr; ///< The copied contacts on the host memory
		int *numContacts = nullptr; ///< The copied numContacts on the device memory
		float3 *midpoints = nullptr; ///< The midpoints for the segments
	};
}

#endif