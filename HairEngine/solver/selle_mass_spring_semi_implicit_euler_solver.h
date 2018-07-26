#pragma once

#include "../util/mathutil.h"
#include "selle_mass_spring_solver_base.h"

namespace HairEngine {
	/**
	 * Subclass of SelleMassSpringSolver, using semi implicit euler for integration.
	 */
	class SelleMassSpringSemiImplcitEulerSolver: public SelleMassSpringSolverBase {

	HairEngine_Public:

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			SelleMassSpringSolverBase::setup(hair, currentTransform);
			im = new Eigen::Vector3f[nparticle];
		}

		void tearDown() override {
			SelleMassSpringSolverBase::tearDown();
			HairEngine_SafeDeleteArray(im);
		}

	HairEngine_Protected:
		void integrate(Eigen::Vector3f* pos, Eigen::Vector3f* vel, Eigen::Vector3f* outVel, float t) override {

			float f1 = t / pmass;

			// Initialize the impluse
			mapParticle(false, [this](Hair::Particle::Ptr par, size_t i) {
				im[i] = par->impulse;
			});

			mapStrand(false, [this, f1, pos, vel, outVel] (size_t si) {
				// Compute the spring force
				for (auto spIt = springs + springStartIndexForStrand[si]; 
					spIt != springs + springStartIndexForStrand[si] + nparticleInStrand[si]; 
					++spIt) {
					Eigen::Vector3f springForce = MathUtility::massSpringForce(pos[spIt->i1], pos[spIt->i2], spIt->k, spIt->l0);

					im[spIt->i1] += springForce;
					im[spIt->i2] -= springForce;
				}

				// Compute the new velocity
				for (size_t i = particleStartIndexForStrand[si]; i < particleStartIndexForStrand[si] + nparticleInStrand[si]; ++i) {
					outVel[i] = vel[i] + im[i] * f1;
				}
			});
		}

		Eigen::Vector3f *im = nullptr; ///< Impluse buffer
	};
}