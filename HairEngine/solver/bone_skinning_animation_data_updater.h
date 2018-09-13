//
// Created by vivi on 2018/9/13.
//

#pragma once

#include "integration_info.h"
#include "../collision/bone_skinning_animation_data.h"
#include "solver.h"

namespace HairEngine {
	/**
	 * A simple wrapper for BoneSkinningAnimationData to make it act as a solver. It will try to update the geometry
	 * of the bone skinning animation data based on current time. It's worth noting that we use the time (0.0f) as the
	 * initial state, so the first geometry will be updated to the info.t for the first "solve" call. It also act as
	 * a sdf mesh input interface to let the sdf build the signed distance field geometry.
	 */
	class BoneSkinningAnimationDataUpdater: public Solver {

	HairEngine_Public:
		/**
		 * Constructor
		 * @param bkad The BoneSkinningAnimationData pointer
		 */
		BoneSkinningAnimationDataUpdater(BoneSkinningAnimationData *bkad): bkad(bkad), time(0.0f) {};

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {
			updateBoneSkinningForCurrentTime();
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {
			time += info.t;
			updateBoneSkinningForCurrentTime();
		}

	HairEngine_Protected:
		BoneSkinningAnimationData *bkad; ///< The BoneSkinningAnimationData reference
		float time; ///< The total time for simulation

		void updateBoneSkinningForCurrentTime() {
			bkad->update(time);
		}
	};
}
