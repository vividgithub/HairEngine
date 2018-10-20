//
// Created by vivi on 2018/10/5.
//

#pragma once

#ifdef HAIRENGINE_ENABLE_CUDA

#include "../util/cudautil.h"
#include "../precompiled/precompiled.h"
#include "integrator.h"
#include "cuda_based_solver.h"


namespace HairEngine {

	void CudaSegmentMidpointComputer_cudaComputeMidpoint(const float3 *parPoses, const int *parStrandIndices,
			float3 *midpoints, int *segStrandIndices, int numParticle, int numStrand, int wrapSize);

	/**
	 * Compute the midpoint of the hair segments in cuda
	 */
	class CudaSegmentMidpointComputer: public CudaBasedSolver {

	HairEngine_Public:

		CudaSegmentMidpointComputer(int wrapSize = 8):
			CudaBasedSolver(Pos_ | StrandIndex_),
			wrapSize(wrapSize) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {
			CudaBasedSolver::setup(hair, currentTransform);

			midpoints = CudaUtility::allocateCudaMemory<float3>(hair.nsegment);
			segStrandIndices = CudaUtility::allocateCudaMemory<int>(hair.nsegment);
//			lengths = CudaUtility::allocateCudaMemory<float>(hair.nsegment);
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			CudaSegmentMidpointComputer_cudaComputeMidpoint(cmc->parPoses, cmc->parStrandIndices,
					midpoints, segStrandIndices, cmc->numParticle, cmc->numStrand, wrapSize);
		}

		void tearDown() override {
			CudaUtility::deallocateCudaMemory(midpoints);
			cudaFree(segStrandIndices);
//			CudaUtility::deallocateCudaMemory(lengths);
		}

		int wrapSize; ///< The cuda wrap size for each thread block

		/// The midpoint of the segments, note that in order to accelerate the mapping from particle to segment.
		/// For the segment that connects the particle i and i + 1 (global index),
		/// the mid point position is stored in midpoints[i - strandIndex[i]]
		float3 *midpoints = nullptr;

		/// The lengths for the segments, same alignment as the midpoints
//		float *lengths = nullptr;

		/// The segment strand indices, we can query the global index for the first connected by i + segStrandIndices[i]
		int *segStrandIndices = nullptr;
	};
}

#endif
