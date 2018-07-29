#pragma once

#ifdef HAIRENGINE_ENABLE_OPENMP
#include "omp.h"
#endif

#include <functional>

#include "../precompiled/precompiled.h"

namespace HairEngine {
	/**
	 * A container for some parallism function
	 */
	class ParallismUtility {
	HairEngine_Public:
		/**
		 * Parallel for loop for the range [start, end)
		 * 
		 * @param start The start index
		 * @param end The end index (Make sure end > start)
		 * @block The block function which accepts an index for parallel executing
		 */
		static inline void parallelFor(size_t start, size_t end, const std::function<void(size_t)> & block) {
#ifdef HAIRENGINE_ENABLE_OPENMP
			#pragma omp parallel for
			for (size_t i = start; i != end; ++i)
				block(i);
#else
			// If the OpenMp is not included, we will do it sequentially
			for (size_t i = start; i != end; ++i)
				block(i);
#endif
		}

		/**
		 * A useful function wrapper to for conditional parallel executing.
		 * If the "parallel" is assigned to true, then parallsim will be excuted for 
		 * the block function in the range of [start, end). Otherwise, block will do 
		 * sequentially from start to end - 1.
		 * 
		 * @param parallel Whether to enable parallism
		 * @param start The start index 
		 * @param end The end index
		 * @block block The block function for executing
		 */
		static inline void conditionalParallelFor(bool parallel, size_t start, size_t end, const std::function<void(size_t)> & block) {
			if (parallel)
				parallelFor(start, end, block);
			else {
				for (size_t i = start; i != end; ++i)
					block(i);
			}
		}
	};
}
