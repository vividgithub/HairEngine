//
// Created by vivi on 2018/10/7.
//

#pragma once

#include <map>
#include <string>

#include "solver.h"
#include "integrator.h"
#include "cuda_memory_converter.h"

#ifdef HAIRENGINE_ENABLE_CUDA

namespace HairEngine {

	/**
	 * CudaBasedSolver is a based class for all cuda solver. Every subclass could initialize the
	 * CudaBasedSolver with an assert options. An assert options is a guarantee that some memory (like positions,
	 * velocities) has been correctly copied to the device memory by the CudaMemoryConverter. It also checks whether
	 * there's an CudaMemoryConverter before the current solver. It not "CudaMemoryConverter" or the some memory
	 * has not been correctly copied, an exception will be thrown in the "setup stage". The subclass could use
	 * the "cmc" to get the corresponding "CudaMemoryConverter".
	 */
	class CudaBasedSolver: public Solver {
	HairEngine_Public:

		CudaBasedSolver(int assertCopyOptions): assertCopyOptions(assertCopyOptions) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {
			// Pre condition check whether
			static const std::map<int, std::string> copyOptionsToString = {
					{ RestPos_, "\"rest position\"" },
					{ Pos_, "\"position\""},
					{ Vel_, "\"velocity\""},
					{ Impulse_, "\"impulse\""},
					{ LocalIndex_, "\"local index\""},
					{ GlobalIndex_, "\"global index\""},
					{ StrandIndex_, "\"strand index\""}
			};

			// For a cuda-based solver, the integrator must contain the CudaMemoryConverter
			// in front of this solver
			cmc = integrator->rfindSolver<CudaMemoryConverter>(0, solverIndex);
			if (!cmc)
				throw HairEngineInitializationException("[CudaBasedSolver] Unable to find CudaMemoryConverter before in the same integrator");

			// The assertCopyOptions should be the subset of the copy options
			int differenceCopyOptions = assertCopyOptions - (assertCopyOptions & cmc->getCopyOptions());
			if (differenceCopyOptions != 0) {
				// Some has not been copy to the device buffer
				std::vector<int> noCopyOptions;
				for (int i = 0; i < 30; ++i) {
					if (((1 << i) & differenceCopyOptions) != 0)
						noCopyOptions.emplace_back(i);
				}

				// Compose the string
				std::string exceptionMessage = "[CudaBasedSolver] Some memory has not been correctly copied to device memory: ";
				exceptionMessage += copyOptionsToString.at(1 << noCopyOptions[0]);
				for (int i = 1; i < noCopyOptions.size(); ++i)
					exceptionMessage += ", " + copyOptionsToString.at(noCopyOptions[i]);

				throw HairEngineInitializationException(exceptionMessage);
			}
		}

	HairEngine_Protected:
		int assertCopyOptions;
		CudaMemoryConverter *cmc = nullptr; ///< The memory converter in the same integrator
	};
}

#endif