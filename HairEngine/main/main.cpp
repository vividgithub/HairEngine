#include <iostream>
#include <sstream>
#include <fstream>

#include "Eigen/StdVector"

#include "../geo/hair.h"
#include "../solver/integrator.h"
#include "../solver/force_applier.h"
#include "../solver/hair_visualizer.h"
#include "../solver/selle_mass_spring_semi_implicit_euler_solver.h"
#include "../solver/selle_mass_spring_implicit_solver.h"
#include "../solver/selle_mass_spring_visualizer.h"
#include "../solver/selle_mass_spring_implicit_heptadiagnoal_solver.h"
#include "../solver/position_commiter.h"
#include "../solver/signed_distance_field_solid_collision_solver.h"
#include "../solver/segment_knn_solver.h"
#include "../solver/segment_knn_solver_visualizer.h"
#include "../solver/hair_contacts_impulse_solver.h"
#include "../solver/hair_contacts_and_collision_impulse_visualizer.h"
#include "../solver/bone_skinning_animation_data_visualizer.h"
#include "../solver/bone_skinning_animation_data_updater.h"
#include "../solver/sdf_collision_solver.h"

using namespace HairEngine;
using namespace std;
using namespace Eigen;
using std::cout;

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

SelleMassSpringSolverBase::Configuration massSpringCommonConfiguration(
	20000.0f,
	10000.0f,
	10000.0f,
	2000.0f,
	15.0f,
	1.05f,
	4.0f,
	25.0f,
	0.0f
);

void testBoneSkinning() {
	BoneSkinningAnimationData bkad("/Users/vivi/Developer/Project/HairEngine/Houdini/Scenes/Head Rotation 1/Rotation1.bkad");

	Eigen::Affine3f initialBoneTransform = bkad.getRestBoneTransform(0);

	// Generate an empty hair
	//std::vector<int> hairStrandSizes = { };
	//std::vector<Eigen::Vector3f> hairParticlePoses = { };
	//const auto hair = make_shared<Hair>(hairParticlePoses.begin(), hairStrandSizes.begin(), hairStrandSizes.end());
	const auto hair = make_shared<Hair>(Hair("/Users/vivi/Developer/Project/HairEngine/Houdini/Resources/Models/Feamle 04 Retop/Hair/Curly-50000-p25.hair", initialBoneTransform.inverse(Eigen::Affine)));

	cout << "Creating integrator..." << endl;
	Integrator integrator(hair, initialBoneTransform);

	auto gravitySolver = integrator.addSolver<FixedAccelerationApplier>(true, Vector3f(0.0f, -9.81f, 0.0f));

	auto segmentKnnSolver = integrator.addSolver<SegmentKNNSolver>(0.004f);
	auto hairContactsSolver = integrator.addSolver<HairContactsImpulseSolver>(segmentKnnSolver.get(), 0.0040f, 0.012f, 10, 1000.0f);
	//auto collisionImpulseSolver = integrator.addSolver<CollisionImpulseSolver>(segmentKnnSolver.get(), 15, 2500.0f, 6);

	auto massSpringConf = massSpringCommonConfiguration;
	//massSpringConf.maxIntegrationTime = 1.0f / 120.0f;
	auto massSpringSolver = integrator.addSolver<SelleMassSpringImplcitHeptadiagnoalSolver>(massSpringConf);

	auto boneSkinningUpdater = integrator.addSolver<BoneSkinningAnimationDataUpdater>(&bkad);
	auto cudaMemoryConverter = integrator.addSolver<CudaMemoryConverter>(Pos_ | Vel_ | LocalIndex_);

	auto sdfCollisionConf = SDFCollisionConfiguration { {128, 128, 128}, 0.1f, 5, 0.0f, 50.0f, 1e-4f, false };
	auto sdfCollisionSolver = integrator.addSolver<SDFCollisionSolver>(sdfCollisionConf, boneSkinningUpdater.get());

	auto cudaMemoryInverseConverter = integrator.addSolver<CudaMemoryInverseConverter>(cudaMemoryConverter.get());

	// Add visualizer
	auto hairVplyVisualizer = integrator.addSolver<HairVisualizer>(
		R"(/Users/vivi/Desktop/HairData)",
		"TestHair-${F}-Hair.hair",
		1.0 / 24.f,
		nullptr
	);

//	auto hairContactsVisualizer = integrator.addSolver<HairContactsAndCollisionImpulseSolverVisualizer>(
//		R"(/Users/vivi/Desktop/HairContacts)",
//		"TestHair-${F}-HairContacts.vply",
//		1.0 / 24.f,
//		hairContactsSolver.get(),
//		nullptr
//	);

//	auto springVplyVisualizer = integrator.addSolver<SelleMassSpringVisualizer>(
//		R"(/Users/vivi/Desktop/HairData)",
//		"TestHair-${F}-Spring.vply",
//		1.0 / 24.0f,
//		massSpringSolver.get()
//	);
//
//	auto sdfCollisionVisualizer = integrator.addSolver<SDFCollisionVisualizer>(
//			"/Users/vivi/Desktop/BoneSkinning",
//			"${F}.vply",
//			1.0 / 24.0f,
//			sdfCollisionSolver.get()
//	);

	const float gravityScale = 1.0f;
	float particleMass = gravityScale * massSpringSolver->getParticleMass();
	gravitySolver->setMass(&particleMass);
	float simulationTime = 1.0f / 120.0f;

	for (int i = 1; i <= 2500; i += 1) {
		cout << "Simulation Frame " << i << "..." << endl;

		Eigen::Affine3f currentBoneTransform = bkad.getBoneTransform(0, i * simulationTime);
		integrator.simulate(simulationTime, currentBoneTransform);

//		if (i > 500) {
//			particleMass = massSpringSolver->getParticleMass();
//			gravitySolver->setMass(&particleMass);
//		}
	}
}

int main() {
//	testOpenMPEnable();
//	validSolverCorretness();
	testBoneSkinning();
	return 0;
}
