#pragma once

#ifdef HAIRENGINE_ENABLE_OPENMP
#include "omp.h"
#endif

#include <functional>
#include <thread>

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
		 * Get the maximum thread to maximize the computation. If OpenMP is not included,
		 * return 1.
		 */
		static inline unsigned getOpenMPMaxHardwareConcurrency() {
#ifdef HAIRENGINE_ENABLE_OPENMP
			const auto maxThreadNumber = std::thread::hardware_concurrency();
			return maxThreadNumber > 0 ? maxThreadNumber : 1;
#else
			return 1;
#endif
		}

		/**
		 * Parallel for loop for the range [start, end), and with additional thread index
		 * 
		 * @param start The start index of the for loop
		 * @param end The end index for the loop
		 * @param block The executing block function which accepts a loop index i and an threadID, which indicating the thread number
		 */
		static inline void parallelForWithThreadIndex(size_t start, size_t end, const std::function<void(size_t, int)> & block) {
#ifdef HAIRENGINE_ENABLE_OPENMP
			static unsigned maxThreadNumber = 0;

			// Only executing once
			if (maxThreadNumber == 0) {
				maxThreadNumber = getOpenMPMaxHardwareConcurrency();
			}

			#pragma omp parallel num_threads(maxThreadNumber)
			{
				int threadID = omp_get_thread_num();
				#pragma omp for
				for (size_t i = start; i != end; ++i)
					block(i, threadID);
			}
#else
			for (size_t i = start; i < end; ++i)
				block(i, 0);
#endif
		}

		/**
		 * Conditional parallel for for "parllelForWithThreadIndex", we use a predicate to indicate whether to 
		 * do it in parallel or sequential
		 * 
		 * @param parallel Whether to do the for loop in paralllel
		 * @param start The start index of the for loop
		 * @param end The end index of the for loop
		 * @param block The executing block function
		 */
		static inline void conditionalParallelForWithThreadIndex(bool parallel, size_t start, size_t end, const std::function<void(size_t, int)> & block) {
			if (parallel)
				parallelForWithThreadIndex(start, end, block);
			else {
				for (size_t i = start; i != end; ++i)
					block(i, 0); // We use thread 0 
			}
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
