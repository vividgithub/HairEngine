#pragma once

#include "../util/mathutil.h"
#include "selle_mass_spring_solver_base.h"

namespace HairEngine {
	/**
	 * Subclass of SelleMassSpringSolver, using semi implicit euler for integration.
	 */
	class SelleMassSpringSemiImplcitEulerSolver: public SelleMassSpringSolverBase {

	HairEngine_Public:

		using SelleMassSpringSolverBase::SelleMassSpringSolverBase; // Inherit constructor

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			SelleMassSpringSolverBase::setup(hair, currentTransform);
			im = new Eigen::Vector3f[nparticle];
		}

		void tearDown() override {
			HairEngine_SafeDeleteArray(im);
			SelleMassSpringSolverBase::tearDown();
		}

	HairEngine_Protected:
		void integrate(Eigen::Vector3f* pos, Eigen::Vector3f* vel, Eigen::Vector3f* outVel, const IntegrationInfo &info) override {

			float f1 = info.t / pmass;

			// Initialize the impluse
			mapParticle(true, [this, vel](Hair::Particle::Ptr par, int i) {
				im[i] = par->impulse - damping * vel[i];
			});

			mapStrand(true, [this, f1, pos, vel, outVel, &info] (int si) {
				// Compute the spring force
				for (auto spIt = springs + springStartIndexForStrand[si]; 
					spIt != springs + springStartIndexForStrand[si] + nspringInStrand[si]; 
					++spIt) {
					Eigen::Vector3f springForce = MathUtility::massSpringForce(pos[spIt->i1], pos[spIt->i2], spIt->k, spIt->l0);

					im[spIt->i1] += springForce;
					im[spIt->i2] -= springForce;
				}

				// Compute the new velocity
				for (size_t i = particleStartIndexForStrand[si] + 1; i < particleStartIndexForStrand[si] + nparticleInStrand[si]; ++i) {
					outVel[i] = vel[i] + im[i] * f1;
				}

				// The hair root velocity
				const size_t & ri = particleStartIndexForStrand[si]; // Strand root index
				auto pr = p(ri);

				// Compute the hair root velocity
				outVel[particleStartIndexForStrand[si]] = (info.transform * pr->restPos - info.previousTransform * pr->restPos) / info.t;
			});
		}

		Eigen::Vector3f *im = nullptr; ///< Impluse buffer
	};
}