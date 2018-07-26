#include <iostream>

#include "../geo/hair.h"
#include "../solver/integrator.h"
#include "../solver/hair_visualizer.h"
#include "../solver/selle_mass_spring_semi_implicit_euler_solver.h"

int main() {
	using namespace HairEngine;
	using namespace std;

	cout << "Reading Hair" << endl;
	auto hair = std::make_shared<Hair>(Hair("C:\\Users\\VividWinPC1\\Developer\\Project\\HairEngine\\Houdini\\Resources\\Models\\Feamle 04 Retop\\Hair\\Straight-50000.hair").resample(537));

	Integrator integrator(hair, Eigen::Affine3f::Identity());

	auto massSpringSolver = integrator.addSolver<SelleMassSpringSemiImplcitEulerSolver>();
	integrator.addSolver<HairVisualizer>(R"(C:\Users\VividWinPC1\Desktop)", "TestHair-${F}.vply", massSpringSolver.get());

	cout << "Simulate" << endl;

	integrator.simulate(0.03f, Eigen::Affine3f::Identity());

	cout << "Done" << endl;

	//hair->writeToFile(R"(C:\Users\VividWinPC1\Desktop\SampledHair.hair)");

	return 0;
}