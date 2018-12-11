//
// Created by vivi on 2018/10/15.
//

#pragma once

#ifdef HAIRENGINE_ENABLE_CUDA

#include <HairEngine/HairEngine/geo/hair.h>
#include "../accel/particle_spatial_hashing.h"
#include "cuda_segment_midpoint_computer.h"
#include "visualizer.h"

#include "VPly/vply.h"

namespace HairEngine {
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
	);

	class CollisionImpulseCudaSolver: public CudaBasedSolver {

		friend class CollisionImpulseCudaVisualizer;

	HairEngine_Public:

		CollisionImpulseCudaSolver(CudaSegmentMidpointComputer *smc, int maxCollisionsPerSegment,
				int maxForceCountPerSegment, float kCollisionSpring, float resolution, int wrapSize = 8):
			CudaBasedSolver(Pos_ | Vel_ | Impulse_),
			smc(smc),
			maxCollisions(maxCollisionsPerSegment),
			maxForceCount(maxForceCountPerSegment),
			k(kCollisionSpring),
			resolution(resolution),
			wrapSize(wrapSize) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {
			CudaBasedSolver::setup(hair, currentTransform);

			int n = maxCollisions * hair.nsegment;

			collisions = CudaUtility::allocateCudaMemory<int>(n);
			cpas = CudaUtility::allocateCudaMemory<float2>(n);
			dns = CudaUtility::allocateCudaMemory<float3>(n);
			l0s = CudaUtility::allocateCudaMemory<float>(n);

			numCollisions = CudaUtility::allocateCudaMemory<int>(hair.nsegment);

			segForces = CudaUtility::allocateCudaMemory<float3>(hair.nsegment);
			numSegForces = CudaUtility::allocateCudaMemory<int>(hair.nsegment);

			// Compute the average length
			// Set up the Particle spatial hashing
			float radius = 0.0f;
			for (int i = 0; i < hair.nsegment; ++i) {
				radius += (hair.segments[i].p1->restPos - hair.segments[i].p2->restPos).norm();
			}
			radius /= (hair.nsegment * 2.0f);

			psh = new ParticleSpatialHashing(hair.nsegment, make_float3(radius / resolution));

			// Fill the numCollisons
			cudaMemset(numCollisions, 0x00, sizeof(int) * hair.nsegment);
		}

		void tearDown() override {
			CudaUtility::deallocateCudaMemory(collisions);
			CudaUtility::deallocateCudaMemory(cpas);
			CudaUtility::deallocateCudaMemory(dns);
			CudaUtility::deallocateCudaMemory(l0s);
			CudaUtility::deallocateCudaMemory(numCollisions);

			delete psh;
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			Solver::solve(hair, info);

			psh->update(smc->midpoints);

			auto startTime = std::chrono::high_resolution_clock::now();

			CollisionImpulseCudaSolver_updateCollisionSpring(
					psh->hashParStartsDevice,
					psh->hashParEndsDevice,
					psh->pidsDevice,
					cmc->parPoses,
					cmc->parVels,
					cmc->parImpulses,
					smc->segStrandIndices,
					collisions,
					cpas,
					dns,
					l0s,
					numCollisions,
					segForces,
					numSegForces,
					hair.nsegment,
					hair.nstrand,
					maxCollisions,
					maxForceCount,
					k,
					psh->dInv,
					psh->numHashShift,
					info.t,
					wrapSize
			);

			auto endTime = std::chrono::high_resolution_clock::now();

			auto diffInUs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
			printf("[CollisionImpulseCudaSolver] Timing: %lld ms(%lld us)\n", diffInUs.count() / 1000, diffInUs.count());
		}

	HairEngine_Protected:
		CudaSegmentMidpointComputer *smc; ///< The midpoint reference
		ParticleSpatialHashing *psh = nullptr;

		int maxCollisions; ///< The max collisions per segment
		int maxForceCount; ///< The maximum force count

		float k; ///< The collision stiffness

		// Particle spatial hashing options
		float resolution; ///< The resolution of the grid to build
 		int wrapSize; ///< The CUDA computation kernel wrap size

 		// Segment collision info
 		/// The segment id for another collision pair, size equals to "maxCollisions * numSegment"
 		int *collisions = nullptr;

 		/// Stores the closest point approach (CPA) t1, and t2 and every collision spring, size equals to
 		/// "maxCollisions * numSegment"
 		float2 *cpas = nullptr;

 		/// The normalized direction for the collision spring, size equals to "maxCollisions * numSegment"
 		float3 *dns = nullptr;

 		/// The rest length for the collision spring, size equals to "maxCollisions * numSegment"
 		float *l0s = nullptr;

 		/// Capture how many collision spring have been created, size equals to "numSegment"
 		int *numCollisions = nullptr;

 		/// The total segment force for each iteration, size equal to "numSegment"
 		float3 *segForces = nullptr;

 		/// The number of segment forces have been accumulated into the "segForces" array, size equal to "numSegForces"
 		int *numSegForces = nullptr;
	};

	class CollisionImpulseCudaVisualizer: public Visualizer {

	HairEngine_Public:

		CollisionImpulseCudaVisualizer(
				const std::string &directory,
				const std::string &filenameTemplate,
				float timestep,
				CollisionImpulseCudaSolver *collisionImpulseCudaSolver
		):
				Visualizer(directory, filenameTemplate, timestep),
				cics(collisionImpulseCudaSolver) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {

			if (!cics)
				return;

			int n = cics->maxCollisions * hair.nsegment;

			collisions = new int[n];
			cpas = new float2[n];
			dns = new float3[n];
			l0s = new float[n];

			numCollisions = new int[hair.nsegment];

			segForces = new float3[hair.nsegment];
			numSegForces = new int[hair.nsegment];
		}

		void tearDown() override {
			delete [] collisions;
			delete [] cpas;
			delete [] dns;
			delete [] l0s;
			delete [] numCollisions;
			delete [] segForces;
			delete [] numSegForces;
		}

		void visualize(std::ostream &os, Hair &hair, const IntegrationInfo &info) override {

			if (!cics)
				return;

			int n = cics->maxCollisions * hair.nsegment;

			// Copy the data from device to hosts
			CudaUtility::copyFromDeviceToHost(collisions, cics->collisions, n);
			CudaUtility::copyFromDeviceToHost(cpas, cics->cpas, n);
			CudaUtility::copyFromDeviceToHost(dns, cics->dns, n);
			CudaUtility::copyFromDeviceToHost(l0s, cics->l0s, n);
			CudaUtility::copyFromDeviceToHost(numCollisions, cics->numCollisions, hair.nsegment);
//			CudaUtility::copyFromDeviceToHost(segForces, cics->segForces, hair.nsegment);
//			CudaUtility::copyFromDeviceToHost(numSegForces, cics->numSegForces, hair.nsegment);

			// Visualize
			for (int idx1 = 0; idx1 < hair.nsegment; ++idx1) if (numCollisions[idx1] > 0) {

				auto li1 = idx1 / hair.nstrand;
				auto si1 = idx1 % hair.nstrand;
				auto seg1 = hair.strands[si1].segmentInfo.beginPtr + li1;
				auto sid1 = seg1->p1->globalIndex - seg1->p1->strandIndex;

				int go = idx1 * cics->maxCollisions;

				for (int i = 0; i < numCollisions[idx1]; ++i) {

					const int idx2 = collisions[go + i];
					auto li2 = idx2 / hair.nstrand;
					auto si2 = idx2 % hair.nstrand;
					auto seg2 = hair.strands[si2].segmentInfo.beginPtr + li2;
					auto sid2 = seg2->p1->globalIndex - seg2->p1->strandIndex;

					const float2 & t = cpas[go + i];
					const float3 & dn = dns[go + i];
					const float & l0 = l0s[go + i];

					Eigen::Vector3f s1p = MathUtility::lerp(seg1->p1->pos, seg1->p2->pos, t.x);
					Eigen::Vector3f s2p = MathUtility::lerp(seg2->p1->pos, seg2->p2->pos, t.y);

					VPly::writeLine(
							os,
							EigenUtility::toVPlyVector3f(s1p),
							EigenUtility::toVPlyVector3f(s2p),
							VPly::VPlyIntAttr("fromid", sid1),
							VPly::VPlyIntAttr("toid", sid2),
							VPly::VPlyIntAttr("sid1", idx1),
							VPly::VPlyIntAttr("sid2", idx2),
//							VPly::VPlyVector2fAttr("t", { t.x, t.y }),
//							VPly::VPlyVector3fAttr("dn", { dn.x, dn.y, dn.z }),
							VPly::VPlyFloatAttr("l0", l0),
							VPly::VPlyFloatAttr("l", (s2p - s1p).norm())
					);
				}
			}
		}

	HairEngine_Protected:

		CollisionImpulseCudaSolver *cics = nullptr;

		// All the variable are in the the host buffer copy from the original device memory with the same name
		int *collisions = nullptr;
		float2 *cpas = nullptr;
		float3 *dns = nullptr;
		float *l0s = nullptr;
		int *numCollisions = nullptr;
		float3 *segForces = nullptr;
		int *numSegForces = nullptr;
	};
}

#endif