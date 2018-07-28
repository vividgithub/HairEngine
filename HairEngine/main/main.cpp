#include <iostream>

#include "../geo/hair.h"
#include "../solver/integrator.h"
#include "../solver/force_applier.h"
#include "../solver/hair_visualizer.h"
#include "../solver/selle_mass_spring_semi_implicit_euler_solver.h"
#include "../solver/selle_mass_spring_visualizer.h"
#include "../solver/position_commiter.h"

int main() {
	using namespace HairEngine;
	using namespace std;

	int simulationCount = 100;
	float simulationStep = 0.01f;
	float totalSimulationTime = simulationStep * simulationCount;
	float integrationStep = 5e-7f;

	cout << "Reading Hair" << endl;
	auto hair = std::make_shared<Hair>(Hair("C:\\Users\\VividWinPC1\\Developer\\Project\\HairEngine\\Houdini\\Resources\\Models\\Feamle 04 Retop\\Hair\\Straight-50000.hair").resample(12351));

	Integrator integrator(hair, Eigen::Affine3f::Identity());

	integrator.addSolver<FixedForceApplier>(true, Eigen::Vector3f(0.0f, -9.81f, 0.0f));
	auto massSpringSolver = integrator.addSolver<SelleMassSpringSemiImplcitEulerSolver>(
		500000.0f, // Stretch stiffness
		200000.0f, // Bending stiffness
		200000.0f, // Torsion stiffness
		15.0f, // Damping
		4.0f, // Colinear max degree
		true, // Enable strain limiting
		25.0f // Mass
	);
	integrator.addSolver<PositionCommiter>();
	integrator.addSolver<HairVisualizer>(R"(C:\Users\VividWinPC1\Desktop\HairData)", "TestHair-${F}-Hair.vply", simulationStep, massSpringSolver.get());
	integrator.addSolver<SelleMassSpringVisualizer>(R"(C:\Users\VividWinPC1\Desktop\HairData)", "TestHair-${F}-Selle.vply", simulationStep, massSpringSolver.get());

	cout << "Simulate" << endl;

	for (size_t i = 0; i < simulationCount; ++i) {
		cout << "Simulating " << i << endl;
		for (float t = 0.0f; t < .995f * simulationStep; t += integrationStep) {
			integrator.simulate(integrationStep, Eigen::Affine3f::Identity());
		}
	}

	cout << "Done" << endl;

	return 0;
}