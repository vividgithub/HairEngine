//
// Created by vivi on 19/06/2018.
//


#pragma once

#include "../geo/hair.h"
#include "../precompiled/precompiled.h"

namespace HairEngine {

	struct IntegrationInfo;

	/**
	 * Base class of the solver class. A solver could be viewed as a function which accepts several configuration
	 * and modify the hair geometry in the given time step. For example, a gravity solver tries to add "gravity impulse"
	 * to the impulse property of particle's impulse. A solver has three typical function, a setup(const Hair &) which
	 * uses to setup the solver for a specific hair geometry. Helper data structures or additional information could be
	 * created in this step. On the hand, the tearDown() method is used to deallocate the additional space. And most
	 * importantly, the "solve(const Hair &, const ItergrationInfo &)" to simulate the behavior of the solver.
	 */
	class Solver {
	HairEngine_Public:
		/**
		 * The setup function. You should allocate some space for storing some additional information that is useful
		 * for the solver.
		 *
		 * @param hair The hair geometry. You could get the particle size or strand size from the hair geometry.
		 */
		virtual void setup(const Hair & hair) {}

		/**
		 * Tear down function. Called when the simulation is done. You should deallocate the space created in the
		 * setup function.
		 */
		virtual void tearDown() {}

		/**
		 * The core implementation of the solver. Override the defualt implementation and modify the hair geometry
		 * in this method.
		 *
		 * @param hair The hair geometry to modify.
		 * @param info The additional information (like the simulation timesteps, the affine transformation ...)
		 */
		virtual void solve(Hair & hair, const IntegrationInfo & info) const {}
	};
}
