#pragma once

#include "selle_mass_spring_solver_base.h"

namespace HairEngine {
	/**
	 * Subclass of SelleMassSpringSolver, using semi implicit euler for integration.
	 */
	class SelleMassSpringSemiImplcitEulerSolver: public SelleMassSpringSolverBase {

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			SelleMassSpringSolverBase::setup(hair, currentTransform);
			HairEngine_AllocatorAllocate(im, nparticle);
		}

		void tearDown() override {
			SelleMassSpringSolverBase::tearDown();
			HairEngine_AllocatorDeallocate(im, nparticle);
		}

	HairEngine_Protected:
		void integrate(Eigen::Vector3f* pos, Eigen::Vector3f* vel, Eigen::Vector3f* dv, float t) override {
			// Initialize the impluse
			mapParticle(false, [this](Hair::Particle::Ptr par, size_t i) {
				im[i] = par->impulse;
			});
		}

		Eigen::Vector3f *im = nullptr; ///< Impluse buffer
	};
}