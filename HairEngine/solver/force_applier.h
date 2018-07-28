#pragma once

#include <memory>

#include "solver.h"

namespace HairEngine {

	/**
	 * Force applier acts as a role for applying force to each 
	 * particles.
	 */
	class ForceApplier: public Solver {

	HairEngine_Public:

		/**
		 * Constructor
		 * 
		 * @param initial If true, it will initialize the particle's "impluse" property, otherwise 
		 * if will add the force to the "impluse" property.
		 */
		ForceApplier(bool initial): initial(initial) {}

		void solve(Hair& hair, const IntegrationInfo& info) override {
			mapParticle(true, [this, &hair](Hair::Particle::Ptr par) {
				const Eigen::Vector3f force = getForce(hair, par);
				if (initial)
					par->impulse = force;
				else
					par->impulse += force;
			});
		}

		/**
		 * Get the force with corresponding hair and its particle
		 * 
		 * @param hair The hair geometry
		 * @param par The particle in the hair
		 */
		virtual Eigen::Vector3f getForce(Hair & hair, Hair::Particle::Ptr par) const = 0;

	HairEngine_Protected:
		bool initial;
	};

	/**
	 * The force applier which apply force equally to each particles, it is useful for 
	 * simulating some fixed field like gravity
	 */
	class FixedForceApplier: public ForceApplier {

	HairEngine_Public:
		/**
		 * Constructor
		 * 
		 * @param initial Whether the force is used to initialize the particle "impulse" 
		 * property
		 * @param fixedForce The force that applies to particles
		 */
		FixedForceApplier(bool initial, Eigen::Vector3f fixedForce) :
			ForceApplier(initial) , fixedForce(std::move(fixedForce)) {}

		Eigen::Vector3f getForce(Hair& hair, Hair::Particle::Ptr par) const override {
			return fixedForce;
		}

	HairEngine_Protected:
		Eigen::Vector3f fixedForce;
	};
}