//
// Created by vivi on 19/06/2018.
//


#pragma once
#include <Eigen/Eigen>
#include <vector>

#include "../precompiled/precompiled.h"
#include "../util/mathutil.h"

namespace HairEngine {
	/**
	 * The integration info stores some needed information for the solver, for example, the affine transform
	 * for the head, the simulation time step, or the previous affine transform.
	 */
	struct IntegrationInfo {
		union {
			float t; ///< Simulation time, short definition
			float simulationTime; ///< Simulation time, long definition
		};

		union {
			Eigen::Affine3f tr; ///< Affine transform for current simulation step, short definition
			Eigen::Affine3f transform; ///< Affine transform for current simulation step, long definition
		};

		union {
			Eigen::Affine3f ptr; ///< Previous affine transform, short definitoin
			Eigen::Affine3f previousTransform; ///< Previous affine transofrm, long definition
		};

		/**
		 * Constructor
		 */
		IntegrationInfo(float simulationTime, Eigen::Affine3f transform, Eigen::Affine3f previousTransform):
			t(simulationTime), tr(transform), ptr(previousTransform) {}

		/**
		 * The integration info can be seem as an interval for simulation. It can be divided the whole simulation interval 
		 * into several small simulation interval. We use two floats t1 and t2 (0.0 <= t1 < t2 <= 1.0) to indicate the simulation 
		 * interval. In another word, we can think of the "this" IntegrationInfo is represented by an interval [0.0, 1.0], we want to 
		 * get the simulationTime, transform, and previousTransform properties for the Integration Info defined by a sub-interval [t1, t2].
		 * We suppose the transform from previous to current transform is s-linear.
		 * 
		 * @param t1 The starting time (0.0 <= t1)
		 * @param t2 The ending time (t1 < t2 <= 1.0)
		 */
		IntegrationInfo lerp(float t1, float t2) {
			HairEngine_DebugAssert(0.0 <= t1 && t1 < t2 && t2 <= 1.0);

			auto lerpedPreviousTransform = MathUtility::lerp(t1, ptr, tr);
			auto lerpedTransform = MathUtility::lerp(t2, ptr, tr);

			return IntegrationInfo((t2 - t1) * t, lerpedTransform, lerpedPreviousTransform);
		}

		/**
		 * Same as the lerp with two parameters with t1 and t2. But instead of just crating one simulation interval, 
		 * we create several splited by the "array" defined by begin and end, which means we yield several interval [*begin, *(begin + 1)],
		 * ..., [*(end - 2), *(end - 1)]
		 * 
		 * @param begin The begin iterator which yield float value
		 * @param end The end iterator which yield float value
		 */
		template <typename FloatCastableIterator>
		std::vector<IntegrationInfo> lerp(FloatCastableIterator begin, FloatCastableIterator end) {
			auto preT = *begin;
			auto preTransform = MathUtility::lerp(preT, ptr, tr);

			std::vector<IntegrationInfo> infos;
			for (auto p = begin + 1; p != end; ++p) {
				auto currentT = *p;
				auto currentTransform = MathUtility::lerp(currentT, ptr, tr);

				// Construct an IntegrationInfo
				infos.emplace_back((currentT - preT) * t, preTransform, currentTransform);

				preT = currentT;
				preTransform = currentTransform;
			}
		}
	};
}