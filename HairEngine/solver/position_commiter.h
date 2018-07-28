#pragma once

#include "integration_info.h"
#include "solver.h"

namespace HairEngine {
	/**
	 * Write the position pos += vel * t to the particles
	 */
	class PositionCommiter: public Solver {
		void solve(Hair& hair, const IntegrationInfo& info) override {
			for (auto p = hair.particles; p != hair.particleEnd(); ++p) {
				p->pos += p->vel * info.t;
			}
		}
	};
}