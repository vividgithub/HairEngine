//
// Created by vivi on 2018/9/20.
//

#pragma once

#ifdef HAIRENGINE_ENABLE_CUDA

#include <vector>
#include <cuda_runtime.h>
#include <type_traits>
#include <algorithm>
#include <functional>

namespace HairEngine {

	struct Identity {
		template<typename U>
		constexpr auto operator()(U&& v) const noexcept
		-> decltype(std::forward<U>(v))
		{
			return std::forward<U>(v);
		}
	};

	namespace CudaUtility {

		/**
		 * Allocate cuda device memory for a specified size
		 * @tparam T The type for the allocated pointer
		 * @param size The size of the array
		 * @return A pointer specifying the allocated memory
		 */
		template <typename T>
		inline T *allocateCudaMemory(int size) {
			T *ret;
			cudaMalloc(&ret, sizeof(T) * size);

			return ret;
		}

		/**
		 * Deallocate the memory for a array
		 * @tparam T The type of the array
		 * @param array The array to be deallocated
		 */
		template <typename T>
		inline void deallocateCudaMemory(T *array) {
			cudaFree(array);
		}

		/**
		 * Safe deallocate the memory for a array
		 * @tparam T The type of the array
		 * @param array The array to be deallocated
		 */
		template <typename T>
		inline void safeDeallocateCudaMemory(T *array) {
			if (array)
				deallocateCudaMemory(array);
		}

		/**
		 * Copy the memory from host to device
		 * @tparam T The type of the copied source and destination pointer
		 * @param dst The destination pointer
		 * @param src The source pointer
		 * @param n The size to copy
		 */
		template <typename T>
		inline void copyFromHostToDevice(T *dst, T *src, int n) {
			cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToDevice);
		}

		/**
		 * Copy the memory from device to host
		 * @tparam T The type of the copied source and destination pointer
		 * @param dst The destination pointer
		 * @param src The source pointer
		 * @param n The size to copy
		 */
		template <typename T>
		inline void copyFromDeviceToHost(T *dst, T *src, int n) {
			cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost);
		}

		/**
		 * Copy a group of data which is defined by a iterator range to a device memory.
		 *
		 * @tparam T The type for the destination (device) pointer
		 * @tparam Iterator The host memory iterator which *Iterator will yield type "T" or a type which can be cast to "T" by TransformOp
		 * @tparam TransformOp A callable type which transform the (*Iterator) to a convertable type for T
		 * @param dst The device memory to copy to
		 * @param begin The source begin iterator
		 * @param end The source end iterator
		 * @param intermediator A host pointer memory with the type T. If the intermediator is not null,
		 * the data defined in the begin and end range will first copy to the intermediator and then pass to the device
		 * memory. Otherwise some temporary memory will be allocated in the host.
		 * @param transform The transform operator
		 */
		template <typename T, class Iterator, class TransformOp = Identity>
		inline void copyFromHostToDevice(T *dst, const Iterator & begin, const Iterator & end,
				T *intermediator = nullptr, TransformOp transform = TransformOp()) {
			// Create a vector to copy to
			if (intermediator) {
				int size = 0;
				for (auto it (begin); it != end; ++it)
					intermediator[size++] = transform(*it);
				copyFromHostToDevice(dst, intermediator, size);
			}
			else {
				// No intermediator, create a vector
				std::vector<T> tempBuffer;
				for (auto it (begin); it != end; ++it)
					tempBuffer.emplace_back(transform(*it));
				copyFromHostToDevice(dst, &(tempBuffer[0]), static_cast<int>(tempBuffer.size()));
			}
		}

		/**
		 * Get the block size and thread size based on current computation size n and the wrap size
		 * @param n The computation size
		 * @param wrapSize The size of the wrap
		 * @param outNumBlock The number of the block
		 * @param outNumThread The number of the thread
		 */
		inline void getGridSizeForKernelComputation(int n, int wrapSize, int *outNumBlock, int *outNumThread) {
			*outNumThread = 32 * wrapSize;
			*outNumBlock = (n + *outNumThread - 1) / *outNumThread;
 		}
	}
}

#endif
