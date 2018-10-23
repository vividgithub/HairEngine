#pragma once

#include "omp.h"

#include <functional>
#include <thread>

#include "../precompiled/precompiled.h"

namespace HairEngine {
	/**
	 * A container for some parallism function
	 */
	class ParallismUtility {
	HairEngine_Public:

		class Spinlock
		{
		HairEngine_Public:

			void lock()
			{
				while (m_lock.test_and_set(std::memory_order_acquire));
			}

			void unlock()
			{
				m_lock.clear(std::memory_order_release);
			}

			Spinlock() = default;
			Spinlock(Spinlock const& other) {};
			Spinlock& operator=(Spinlock const& other) { return *this; }

		HairEngine_Private:

			std::atomic_flag m_lock = ATOMIC_FLAG_INIT;
		};

		/**
		 * Parallel for loop for the range [start, end)
		 * 
		 * @param start The start index
		 * @param end The end index (Make sure end > start)
		 * @block The block function which accepts an index for parallel executing
		 */
		static inline void parallelFor(int start, int end, const std::function<void(int)> & block) {
#ifdef NDEBUG
			#pragma omp parallel for
			for (int i = start; i < end; ++i)
				block(i);
#else
			for (int i = start; i < end; ++i)
				block(i);
#endif
		}

		/**
		 * Get the maximum thread to maximize the computation. If OpenMP is not included,
		 * return 1.
		 */
		static inline unsigned getOpenMPMaxHardwareConcurrency() {
#ifdef NDEBUG
			const auto maxThreadNumber = std::thread::hardware_concurrency();
			return maxThreadNumber > 0 ? maxThreadNumber : 1;
#elif
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
		static inline void parallelForWithThreadIndex(int start, int end, const std::function<void(int, int)> & block) {
#ifdef NDEBUG
			static unsigned maxThreadNumber = 0;

			// Only executing once
			if (maxThreadNumber == 0) {
				maxThreadNumber = getOpenMPMaxHardwareConcurrency();
			}

			#pragma omp parallel num_threads(maxThreadNumber)
			{
				const int threadID = omp_get_thread_num();
				#pragma omp for
				for (int i = start; i < end; ++i)
					block(i, threadID);
			}
#else
			for (int i = start; i < end; ++i)
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
		static inline void conditionalParallelForWithThreadIndex(bool parallel, int start, int end, const std::function<void(int, int)> & block) {
			if (parallel)
				parallelForWithThreadIndex(start, end, block);
			else {
				for (int i = start; i != end; ++i)
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
		static inline void conditionalParallelFor(bool parallel, int start, int end, const std::function<void(int)> & block) {
			if (parallel)
				parallelFor(start, end, block);
			else {
				for (int i = start; i != end; ++i)
					block(i);
			}
		}
	};
}
