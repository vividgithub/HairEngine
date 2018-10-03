//
// Created by vivi on 2018/10/3.
//

#pragma once

#include <vector>
#include <algorithm>
#include <utility>
#include <Eigen/Eigen>

#include "../util/mathutil.h"
#include "../precompiled/precompiled.h"

namespace HairEngine {
	/**
	 * Varying Float can be view as a continuous function defined by multiple points.
	 * It's used to initialize some parameter which varies for strands or particle per
	 * strand.
	 */
	class VaryingFloat {
	HairEngine_Public:

		/**
		 * Initialize the varying float based on 2 float iterator, the x iterator specifies the
		 * x positions (within 0.0 to 1.0) and y position specify the values for the corresponding
		 * x positions.
		 * @tparam XIterator The Iterator for x axis, should convertible to float
		 * @tparam YIterator The Iterator for y axis, should convertible to float
		 * @param xBegin The begin of x iterators
		 * @param xEnd The end of x iterators
		 * @param yBegin The begin of y iterators
		 * @param base The multiplier, which the actual value will be "y * base"
		 */
		template <typename XIterator, typename YIterator>
		VaryingFloat(XIterator xBegin, XIterator xEnd, YIterator yBegin, float base=1.0f) {
			auto xit = xBegin;
			auto yit = yBegin;
			for (; xit != xEnd; ++xit, ++yit)
				points.emplace_back(*xit, *yit);

			HairEngine_DebugAssert(points.x() >= 0.0f && points.y() <= 1.0f);

			// Sort with the increasing order of x
			std::sort(points.begin(), points.end(), [](const Eigen::Vector2f & p1, const Eigen::Vector2f &p2){
				return p1.x() < p2.x();
			});
		}

		/**
		 * Get the interpolated value in the position x
		 * @param x The x position.
		 * @return The interpolated value
		 */
		float operator()(float x) const {
			// Currently only linear support
			int l = 0, r = points.size() - 1;
			while (l + 1 < r) {
				int mid = (l + r) / 2;
				if (points[mid].x() <= x)
					l = mid;
				else
					r = mid;
			}

			return MathUtility::lerp(
					points[l].y(),
					points[r].y(),
					(x - points[l].x()) / (points[r].x() - points[l].x())
			);
		}

	HairEngine_Private:

		std::vector<Eigen::Vector2f> points;
	};
}
