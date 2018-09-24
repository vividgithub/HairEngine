//
// Created by vivi on 2018/9/12.
//

#pragma once

#include "VPly/vply.h"

#include "../collision/bone_skinning_animation_data.h"
#include "visualizer.h"

namespace HairEngine {
	class BoneSkinningAnimationDataVisualizer: public Visualizer {
	HairEngine_Public:
		BoneSkinningAnimationDataVisualizer(const std::string & directory, const std::string & filenameTemplate,
				float timestep, BoneSkinningAnimationData *bkad):
				Visualizer(directory, filenameTemplate, timestep), bkad(bkad) {}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			time += info.t;
			Visualizer::solve(hair, info);
		}

		void visualize(std::ostream &os, Hair &hair, const IntegrationInfo &info) override {

			bkad->update(time);

			// Get the point data and dump to the vply file
			for (int i = 0; i < bkad->npoint; ++i) {
				VPly::writePoint(os, EigenUtility::toVPlyVector3f(bkad->poses[i]));
			}
		}

	HairEngine_Protected:
		BoneSkinningAnimationData *bkad; ///< The reference bone skinning data object
		float time = 0.0f; ///< The total time for simulation
	};
}
