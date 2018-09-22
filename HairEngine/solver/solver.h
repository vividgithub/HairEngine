//
// Created by vivi on 19/06/2018.
//


#pragma once

#include <functional>

#include "../geo/hair.h"
#include "../precompiled/precompiled.h"
#include "../util/parallutil.h"

namespace HairEngine {

	struct IntegrationInfo;
	class Integrator;

	/**
	 * Base class of the solver class. A solver could be viewed as a function which accepts several configuration
	 * and modify the hair geometry in the given time step. For example, a gravity solver tries to add "gravity impulse"
	 * to the impulse property of particle's impulse. A solver has three typical function, a setup(const Hair &) which
	 * uses to setup the solver for a specific hair geometry. Helper data structures or additional information could be
	 * created in this step. On the hand, the tearDown() method is used to deallocate the additional space. And most
	 * importantly, the "solve(const Hair &, const ItergrationInfo &)" to simulate the behavior of the solver.
	 */
	class Solver {

		friend class Integrator;

	HairEngine_Public:
		/**
		 * The setup function. You should allocate some space for storing some additional information that is useful
		 * for the solver.
		 *
		 * @param hair The hair geometry. You could get the particle size or strand size from the hair geometry.
		 * @param currentTransform We will perform an affine transform before the setup function called in the Integrator.
		 * So that in the setup of the Solver, the restPos = currentTransform * pos for particles. It's useful to setup 
		 * some new particles' restPos using currentTransform.
		 */
		virtual void setup(const Hair & hair, const Eigen::Affine3f & currentTransform) {}

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
		virtual void solve(Hair & hair, const IntegrationInfo & info) {}

		virtual ~Solver() = default;

	HairEngine_Protected:
		Hair *hair = nullptr; ///< The solved hair reference
		Integrator *integrator = nullptr; ///< The integrator reference
		int solverIndex = -1; ///< The index for the solver in the integrator

		/* Helper function (used in "solve" context) */

		/**
		 * Iterate all the particles and do some stuff specified by the mapper.
		 * 
		 * @param parallel True to enable parallism, otherwise we will do the mapping sequentially.
		 * @param mapper The mapping function that accepts a particle pointer
		 */
		void mapParticle(bool parallel, const std::function<void(Hair::Particle::Ptr)> &mapper) {

			const auto & block = [&mapper, this] (int i) {
				mapper(hair->particles + i);
			};

			ParallismUtility::conditionalParallelFor(parallel, 0, hair->nparticle, block);
		}

		/**
		 * Iterate all the segments and apply mapper function to them
		 * 
		 * @param parallel True to enable parallism, otherwise the segment will be handled sequentially
		 * @param mapper The mapping function
		 */
		void mapSegment(bool parallel, const std::function<void(Hair::Segment::Ptr)> &mapper) {
			const auto & block = [&mapper, this] (int i) {
				mapper(hair->segments + i);
			};

			ParallismUtility::conditionalParallelFor(parallel, 0, hair->nsegment, block);
		}

		/**
		 * Iterate all the strands and do some stuff specified by the mapper
		 * 
		 * @param parallel True to enable parallism, if enabling, it must guarantee that the 
		 * order for mapping will not affect the final result. Otherwise, we will map sequentially.
		 * 
		 * @param mapper The function for mapping
		 */
		void mapStrand(bool parallel, const std::function<void(Hair::Strand::Ptr)> &mapper) {
			const auto & block = [&mapper, this] (int i) {
				mapper(hair->strands + i);
			};

			ParallismUtility::conditionalParallelFor(parallel, 0, static_cast<int>(hair->nstrand), block);
		}
	};
}
