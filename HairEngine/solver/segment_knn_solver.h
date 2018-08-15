#pragma once

#include "CompactNSearch.h"
#include "solver.h"
#include <iostream>

namespace HairEngine {

	/**
	 * Finding k-nearest neighbour for segments. Currently it is a wrapper for the CompactNSearch
	 */
	class SegmentKNNSolver: public Solver {
	HairEngine_Public:

		SegmentKNNSolver(float radius): radius(radius) {}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			Solver::setup(hair, currentTransform);

			nsearch = new CompactNSearch::NeighborhoodSearch(radius);

			// Construct the position buffer
			posBuffer = new CompactNSearch::Real[3 * hair.nsegment];
			nsearch->add_point_set(posBuffer, hair.nsegment);
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {
			std::cout << "SegmentKNNSolver solve..." << std::endl;

			// Copy the pos in Vector3f into the posBuffer
			mapSegment(true, [this](Hair::Segment::Ptr seg) {
				CompactNSearch::Real *posBufferStartPtr = seg->globalIndex * 3 + posBuffer;

				// We don't assume that the Vector3f in Eigen is aligned and directly use the memcpy
				Eigen::Vector3f midpoint = seg->midpoint();
				posBufferStartPtr[0] = midpoint.x();
				posBufferStartPtr[1] = midpoint.y();
				posBufferStartPtr[2] = midpoint.z();
			});

			nsearch->find_neighbors();
		}

		/**
		 * Get the number of neighbour for the segment with the specific index.
		 * Call after "solve"
		 * 
		 * @param index The index for the segment.
		 * @return Number of neighbour for the segment
		 */
		int getNNeighbourForSegment(int index) const {
			const auto & ps = nsearch->point_set(0);
			return ps.n_neighbors(0, index);
		}

		/**
		 * Get the kth neighbour for the segment with the specific index.
		 * Call after "solve"
		 * 
		 * @param index The index for the segment
		 * @param k The kth neighbour for querying
		 * @return The index for the kth neighbour of the segment with "index"
		 */
		int getNeighbourIndexForSegment(int index, int k) const {
			const auto & ps = nsearch->point_set(0);
			return ps.neighbor(0, index, k);
		}

		void tearDown() override {
			delete nsearch;
			delete[] posBuffer;

			Solver::tearDown();
		}

		/**
		 * Get the radius of the searching
		 * 
		 * @return The radius
		 */
		float getRadius() const {
			return radius;
		}

	HairEngine_Protected:
		float radius; ///< The searching radius

		CompactNSearch::NeighborhoodSearch *nsearch = nullptr; ///< The comparct search kernal
		CompactNSearch::Real *posBuffer = nullptr; ///< The position buffer storing the midpoint of the segments
	};
}