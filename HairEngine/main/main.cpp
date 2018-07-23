#include <iostream>

#include "../geo/hair.h"
#include "../solver/integrator.h"
#include "../solver/hair_particle_visualizer.h"

int main() {
	using namespace HairEngine;
	using namespace std;

	cout << "Reading Hair" << endl;
	auto hair = std::make_shared<Hair>("C:\\Users\\VividWinPC1\\Developer\\Project\\HairEngine\\Houdini\\Resources\\Models\\Feamle 04 Retop\\Hair\\Straight-50000.hair");

	Integrator integrator(hair, Eigen::Affine3f::Identity());

	integrator.addSolver<HairParticleVisualizer>("C:\\Users\\VividWinPC1\\Desktop", "Test-${F}.vply");

	cout << "Simulate" << endl;

	integrator.simulate(0.03f, Eigen::Affine3f::Identity());

	cout << "Done" << endl;

	return 0;
}