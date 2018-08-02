#include <iostream>
#include <sstream>
#include <fstream>

#include "../geo/hair.h"
#include "../solver/integrator.h"
#include "../solver/force_applier.h"
#include "../solver/hair_visualizer.h"
#include "../solver/selle_mass_spring_semi_implicit_euler_solver.h"
#include "../solver/selle_mass_spring_implicit_solver.h"
#include "../solver/selle_mass_spring_visualizer.h"
#include "../solver/selle_mass_spring_implicit_heptadiagnoal_solver.h"
#include "../solver/position_commiter.h"

struct TimingSummary {
	int nstrand;
	int nparticle;
	std::string simulatorName;
	double totalTime;
	double averageStrandTime;
	double averageParticleTime;

	TimingSummary(int nstrand, int nparticle, std::string simulatorName, double totalTime):
		nstrand(nstrand), nparticle(nparticle), simulatorName(std::move(simulatorName)), totalTime(totalTime),
		averageStrandTime(totalTime / nstrand), averageParticleTime(averageStrandTime / nparticle) {}
};

int main() {
	using namespace HairEngine;
	using namespace std;
	using std::cout;

	float integrationStep = 5e-3f;

	int strandNumbers[] = { 2500, 5000, 10000, 20000, 30000, 40000, 50000 };
	int particlePerStrandNumbers[] = { 15, 25, 50, 75, 100, 150 };
	vector<string> simulatorNames = { "Conjugate Gradient", "Parallel Conjugate Gradient", "Our Method" };

	SelleMassSpringSolverBase::Configuration conf(
		500000.0f,
		10000.0f,
		10000.0f,
		1000.0f,
		6.0f,
		true,
		4.0f,
		25.0f
	);

	vector<TimingSummary> summaries;

	cout << "OpenMP Testing" << endl;

	#pragma omp parallel num_threads(ParallismUtility::getOpenMPMaxHardwareConcurrency())
	{
		cout << "Thread ID = " << omp_get_thread_num();
	}

	// Write to summary file
	fstream fout(R"(C:\Users\VividWinPC1\Desktop\HairTimeSummaryNew.csv)", ios::out);

	for (auto strandNumber : strandNumbers)
		for (auto particlePerStrandNumber : particlePerStrandNumbers) {
			ostringstream ss;
			ss << R"(C:\Users\VividWinPC1\Developer\Project\HairEngine\Houdini\Resources\Models\Feamle 04 Retop\Hair\)";
			ss << "Curly-" << strandNumber << "-p" << particlePerStrandNumber << ".hair";

			cout << "Reading hair \"" << ss.str() << "\"..." << endl;
			// Initialize the hair
			auto hair = make_shared<Hair>(ss.str());

			cout << "Perform initial simulation using heptadiagnoal solver..." << endl;
			// Simulating 5 frames 
			{
				auto integrator = Integrator(hair, Eigen::Affine3f::Identity());

				auto accelerationApplier = integrator.addSolver<FixedAccelerationApplier>(true, Eigen::Vector3f(0.0, -9.81f, 0.0f));
				auto hairDynamicsSolver = integrator.addSolver<SelleMassSpringImplcitHeptadiagnoalSolver>(conf);
				integrator.addSolver<PositionCommiter>();

				accelerationApplier->setMass(&hairDynamicsSolver->getParticleMass());

				// Integration
				for (size_t i = 0; i < 5; ++i)
					integrator.simulate(integrationStep, Eigen::Affine3f::Identity());
			}

			ss.str("");
			ss.clear();
			ss << R"(C:\Users\VividWinPC1\Desktop\)" << "InitialSimulation-" << strandNumber << "-p" << particlePerStrandNumber << ".hair";
			cout << "Writing initial simulation result to \"" << ss.str() << "\"" << endl;
			hair->writeToFile(ss.str());
			cout << "Done writing..." << endl;

			// Test integration
			for (const auto & simulatorName : simulatorNames) {

				cout << "Test integration time for " << simulatorName << endl;

				auto integrator = Integrator(hair, Eigen::Affine3f::Identity());

				auto accelerationApplier = integrator.addSolver<FixedAccelerationApplier>(true, Eigen::Vector3f(0.0, -9.81f, 0.0f));
				shared_ptr<SelleMassSpringSolverBase> hairDynamicsSolver = nullptr;
				if (simulatorName == "Conjugate Gradient") {
					hairDynamicsSolver = integrator.addSolver<SelleMassSpringImplicitSolver>(conf, false);
				}
				else if (simulatorName == "Parallel Conjugate Gradient") {
					hairDynamicsSolver = integrator.addSolver<SelleMassSpringImplicitSolver>(conf, true);
				}
				else if (simulatorName == "Our Method") {
					hairDynamicsSolver = integrator.addSolver<SelleMassSpringImplcitHeptadiagnoalSolver>(conf);
				}
				integrator.addSolver<PositionCommiter>();

				accelerationApplier->setMass(&hairDynamicsSolver->getParticleMass());

				// Simulate one frame
				integrator.simulate(integrationStep, Eigen::Affine3f::Identity());

				// Get the timming
				double integrationTime = hairDynamicsSolver->getIntegrationTime();

				// Add to summary
				summaries.emplace_back((int)hairDynamicsSolver->getStrandCount(), (int)hairDynamicsSolver->getParticleCount(), simulatorName, integrationTime);

				cout << "Done integration with time " << integrationTime << endl;

				for (const auto & summary : summaries) {
					fout << summary.simulatorName << "," << summary.nstrand << "," << summary.nparticle
						<< "," << summary.totalTime << "," << summary.averageStrandTime << "," << summary.averageParticleTime << std::endl;
				}
			}
		}

	return 0;
}
