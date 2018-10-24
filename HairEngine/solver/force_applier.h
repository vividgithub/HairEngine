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
	class FixedAccelerationApplier: public ForceApplier {

	HairEngine_Public:
		/**
		 * Constructor
		 * 
		 * @param initial Whether the force is used to initialize the particle "impulse" 
		 * property
		 * @param acceleration The acceleartion that applies to particles
		 */
		FixedAccelerationApplier(bool initial, Eigen::Vector3f acceleration) :
			ForceApplier(initial) , acceleartion(std::move(acceleration)) {}

		/**
		 * Call it before the simulation begins
		 * 
		 * @param massPtr The mass pointer that points to the paticle mass
		 */
		void setMass(float mass) {
			this->mass = mass;
		}

		Eigen::Vector3f getForce(Hair& hair, Hair::Particle::Ptr par) const override {
			return acceleartion * mass;
		}

	HairEngine_Protected:
		Eigen::Vector3f acceleartion; ///< The accleration speed
		float mass; ///< The pointer to the particle mass so that we can get the force
	};
}