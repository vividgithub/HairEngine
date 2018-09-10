#pragma once

namespace CompactNSearch
{
#ifdef USE_DOUBLE
	using Real = double;
#else
	using Real = float;
#endif
	namespace Parallel {
		template <class Iterator, class BlockFunction>
		inline void foreach(const Iterator & begin_, const Iterator & end_, const BlockFunction & block) {
			#pragma omp parallel for
			for (auto it = begin_; it < end_; ++it) {
				block(*it);
			}
		}
	}
}

#define INITIAL_NUMBER_OF_INDICES   50
#define INITIAL_NUMBER_OF_NEIGHBORS 50

#ifdef _MSC_VER
	#include <ppl.h>
#else
	// #include <parallel/algorithm>
#endif
