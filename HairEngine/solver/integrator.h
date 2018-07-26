#pragma once

#include <memory>
#include <vector>
#include <type_traits>

#include "../precompiled/precompiled.h"
#include "../geo/hair.h"
#include "integration_info.h"
#include "solver.h"

namespace HairEngine {

	/**
	 * Integrator uses to combine the solver and the hair geometry together and perform the simulation.
	 * A common Integrator is composed of an instance of hair geometry and some sequential solver. It try 
	 * to call the "setup" function for each added solver in the rest state of hair geometry and pass the IntegrationInfo 
	 * to each solver at the simulation time. 
	 */
	class Integrator {
	HairEngine_Public:
		/**
		 * Constructor
		 * 
		 * @param hairPtr The shared pointer of the simulated hair geometry instance
		 * @param restTransform The initial transform for simulation
		 */
		Integrator(std::shared_ptr<Hair> hairPtr, const Eigen::Affine3f & restTransform)
			: hairPtr(hairPtr), previousTransform(restTransform) {
			
			// Perform a transform to the hair geometry
			for (auto p = hairPtr->particles; p != hairPtr->particleEnd(); ++p) {
				p->pos = restTransform * p->restPos;
			}
		}

		/**
		 * Deconstructor
		 */
		virtual ~Integrator() {
			for (auto solverPtr : solverPtrs)
				solverPtr->tearDown();
		}

		/**
		 * Emplace a solver of SolverSubClass which is initialized with Args... in to the solvers.
		 * 
		 * @param args The arguments that use to initialize the SolverSubClass
		 */
		template <class SolverSubClass, class ...Args, typename = std::enable_if<std::is_base_of<Solver, SolverSubClass>::value>>
		std::shared_ptr<SolverSubClass> addSolver(Args ...args) {
			auto solverPtr = std::make_shared<SolverSubClass>(args...);

			// Call the setup function of the solverPtr
			solverPtr->setup(*hairPtr, previousTransform);

			solverPtrs.push_back(solverPtr);
			return solverPtr;
		}

		/**
		 * Simulate the hair geometry with simulation time t and the final transform
		 */
		void simulate(float t, const Eigen::Affine3f & transform) {
			++currentFrameNumber;

			// Prepare the IntegrationInfo
			IntegrationInfo info(t, transform, previousTransform, currentFrameNumber);

			for (auto solverPtr : solverPtrs) {
				solverPtr->solve(*hairPtr, info);
			}

			previousTransform = transform;
		}

	HairEngine_Protected:
		std::shared_ptr<Hair> hairPtr; ///< The simulated hair geometry

		Eigen::Affine3f previousTransform; ///< The affine transform of previous hair geometry
		size_t currentFrameNumber = 0; ///< Current frame number

		std::vector<std::shared_ptr<Solver>> solverPtrs; ///< The solver pointer that use to guide the simulation
	};
}