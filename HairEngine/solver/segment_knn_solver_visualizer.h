#pragma once

#include "VPly/vply.h"
#include "../util/eigenutil.h"
#include "segment_knn_solver.h"
#include "visualizer.h"

namespace HairEngine {

	/**
	 * The visualization class for SegmentKNNSolver. The visualizer will write the connections as a 
	 * vply line, which connects the midpoint of the two neighbours in the SegmentKNNSolver
	 */
	class SegmentKNNSolverVisualizer: public Visualizer {

	HairEngine_Public:

		/**
		 * Constructor
		 * 
		 * @param segmentKnnSolver The SegmentKNNSolver pointer
		 * @param sampleRate Sometimes it is useless to dump all the connection, we suport to write part of them by using the sampleRate. 
		 * @param directory The directory to write to
		 * @param filenameTemplate The filename template
		 * @param timestep The timestep for writing an vply file
		 */
		SegmentKNNSolverVisualizer(const std::string & directory, const std::string & filenameTemplate, float timestep, SegmentKNNSolver *segmentKnnSolver, int sampleRate = 1):
			Visualizer(directory, filenameTemplate, timestep), segmentKnnSolver(segmentKnnSolver), sampleRate(sampleRate < 1 ? 1 : sampleRate) {}

		void visualize(std::ostream& os, Hair& hair, const IntegrationInfo& info) override {

			int nconnection = 0;
			for (int idx1 = 0; idx1 < hair.nsegment; idx1 += sampleRate) {
				auto seg1 = hair.segments + idx1;

				int nneighbour = segmentKnnSolver->getNNeighbourForSegment(idx1);

				for (int i = 0; i < nneighbour; ++i) {
					int idx2 = segmentKnnSolver->getNeighbourIndexForSegment(idx1, i);
					auto seg2 = hair.segments + idx2;

					if (idx1 < idx2) {
						VPly::writeLine(
							os,
							EigenUtility::toVPlyVector3f(seg1->midpoint()),
							EigenUtility::toVPlyVector3f(seg2->midpoint()),
							VPly::VPlyIntAttr("fromid", seg1->globalIndex),
							VPly::VPlyIntAttr("toid", seg2->globalIndex)
 						);
						++nconnection;
					}
				}
			}

			std::cout << "SegmentKNNSolverVisualizer average connection/particle: " << static_cast<float>(nconnection * 2) / hair.nparticle << std::endl;
		}

	HairEngine_Protected:
		SegmentKNNSolver *segmentKnnSolver; ///< The SegmentKNNSolver pointer
		int sampleRate; ///< The sample rate
	};
}