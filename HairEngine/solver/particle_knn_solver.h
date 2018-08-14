#pragma once

#include "CompactNSearch.h"
#include "solver.h"

namespace HairEngine {

	/**
	 * Finding k-nearest neighbour for particles. Currently it is a wrapper for the CompactNSearch
	 */
	class ParticleKNNSolver: public Solver {
	HairEngine_Public:

		ParticleKNNSolver(float radius): radius(radius) {}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			Solver::setup(hair, currentTransform);

			nsearch = new CompactNSearch::NeighborhoodSearch(radius);

			// Construct the position buffer
			posBuffer = new CompactNSearch::Real[hair.nparticle];
			nsearch->add_point_set(posBuffer, hair.nparticle);
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {
			// Copy the pos in Vector3f into the posBuffer
			mapParticle(true, [this](Hair::Particle::Ptr par) {
				CompactNSearch::Real *posBufferStartPtr = par->globalIndex * 3 + posBuffer;

				// We don't assume that the Vector3f in Eigen is aligned and directly use the memcpy
				posBufferStartPtr[0] = par->pos.x();
				posBufferStartPtr[1] = par->pos.y();
				posBufferStartPtr[2] = par->pos.z();
			});
		}

		/**
		 * Get the number of neighbour for the particle with the specific index.
		 * Call after "solve"
		 * 
		 * @param index The index for the particle.
		 * @return Number of neighbour for the particle
		 */
		int getNNeighbourForParticle(int index) const {
			const auto & ps = nsearch->point_set(0);
			return ps.n_neighbors(0, index);
		}

		/**
		 * Get the kth neighbour for the particle with the specific index.
		 * Call after "solve"
		 * 
		 * @param index The index for the particle
		 * @param k The kth neighbour for querying
		 * @return The index for the kth neighbour of the particle with "index"
		 */
		int getNeighbourIndexForParticle(int index, int k) const {
			const auto & ps = nsearch->point_set(0);
			return ps.neighbor(0, index, k);
		}

		void tearDown() override {
			delete nsearch;
			delete[] posBuffer;

			Solver::tearDown();
		}

	HairEngine_Protected:
		float radius; ///< The searching radius

		CompactNSearch::NeighborhoodSearch *nsearch = nullptr;
		CompactNSearch::Real *posBuffer = nullptr;
	};
}