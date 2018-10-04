#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>

#include "Eigen/StdVector"

#include "cxxopts.hpp"
#include "INIReader.h"

#include "../util/parmutil.h"
#include "../solver/position_commiter.h"
#include "../solver/force_applier.h"
#include "../solver/selle_mass_spring_implicit_heptadiagnoal_solver.h"
#include "../solver/collision_impulse_solver.h"
#include "../solver/bone_skinning_animation_data_updater.h"
#include "../solver/selle_mass_spring_visualizer.h"
#include "../solver/hair_contacts_and_collision_impulse_visualizer.h"

using namespace HairEngine;
using namespace std;
using namespace Eigen;
using std::cout;

/**
 * Get varying float from a string
 * @param s The string
 * @return A varying float variable
 */
VaryingFloat getVaryingFloat(const std::string & s) {
	istringstream is(s);

	std::vector<float> xy;
	while (is) {
		xy.emplace_back();
		is >> xy.back();
	}

	HairEngine_DebugAssert(xy.size() % 2 == 0);

	auto xBegin = xy.begin();
	auto xEnd = xBegin + xy.size() / 2;

	return VaryingFloat(xBegin, xEnd, xEnd);
}

int main(int argc, char **argv) {

	try {
		cxxopts::Options cmdOptions("HairEngine[cmd]", "HairEngine command line tool");

		cmdOptions.add_options()("i,input", "The input \".ini\" file", cxxopts::value<std::string>());
		auto result = cmdOptions.parse(argc, argv);

		if (result.count("input") == 0) {
			std::cout << "Please specify the configuration file" << std::endl;
			return 1;
		}

		auto iniFilePath = result["input"].as<string>();
		INIReader ini(iniFilePath);

		BoneSkinningAnimationData bkad(ini.Get("hair", "bkad_path"));

		Eigen::Affine3f initialBoneTransform = bkad.getRestBoneTransform(0);

		// Generate an empty hair
		int resampleRate = ini.GetBoolean("hair", "enable_resampling") ? ini.GetInteger("hair", "resample_rate") : 1;
		const auto hair = make_shared<Hair>(Hair(ini.Get("hair", "path"),
		                                         initialBoneTransform.inverse(Eigen::Affine)).resample(resampleRate));

		cout << "Creating integrator..." << endl;
		Integrator integrator(hair, initialBoneTransform);

		Eigen::Vector3f gravity = Eigen::Vector3f(
				ini.GetReal("common", "gravityx"),
				ini.GetReal("common", "gravityy"),
				ini.GetReal("common", "gravityz")
		);
		auto gravitySolver = integrator.addSolver<FixedAccelerationApplier>(true, gravity);

		auto enableHairContacts = ini.GetBoolean("haircontacts", "enable");
		auto enableHairCollisions = ini.GetBoolean("haircollisions", "enable");

		HairContactsImpulseSolver *hairContactsSolverPtr = nullptr;
		CollisionImpulseSolver *hairCollisionSolverPtr = nullptr;

		if (enableHairContacts || enableHairCollisions) {
			float knnRadius = 0.0f;
			if (enableHairContacts)
				knnRadius = std::max(knnRadius, ini.GetReal("haircontacts", "creating_distance"));
			if (enableHairCollisions)
				knnRadius = std::max(knnRadius, ini.GetReal("haircollisions", "check_distance"));

			auto segmentKnnSolver = integrator.addSolver<SegmentKNNSolver>(knnRadius);

			if (enableHairContacts) {
				auto hairContactsSolver = integrator.addSolver<HairContactsImpulseSolver>(
						segmentKnnSolver.get(),
						ini.GetReal("haircontacts", "creating_distance"),
						ini.GetReal("haircontacts", "breaking_distance"),
						ini.GetInteger("haircontacts", "max_contacts"),
						ini.GetReal("haircontacts", "stiffness")
				);

				hairContactsSolverPtr = hairContactsSolver.get();
			}

			if (enableHairCollisions) {
				auto hairCollisionSolver = integrator.addSolver<CollisionImpulseSolver>(
						segmentKnnSolver.get(),
						ini.GetInteger("haircollisions", "max_collisions"),
						ini.GetReal("haircollisions", "stiffness"),
						ini.GetInteger("haircollisions", "max_collisions_force_count")
				);

				hairCollisionSolverPtr = hairCollisionSolver.get();
			}
		}

		auto massSpringConf = SelleMassSpringSolverBase::Configuration(
				ini.GetReal("massspring", "stretch_stiffness"),
				ini.GetReal("massspring", "bending_stiffness"),
				ini.GetReal("massspring", "torsion_stiffness"),
				ini.GetReal("massspring", "altitude_stiffness"),
				ini.GetReal("massspring", "damping"),
				getVaryingFloat(ini.Get("massspring", "rigidness")),
				ini.GetReal("massspring", "strain_limiting_tolerance"),
				ini.GetReal("massspring", "colinear_max_degree"),
				ini.GetReal("massspring", "mass"),
				1.0f / ini.GetReal("massspring", "max_integration_fps")
		);

		auto massSpringSolver = integrator.addSolver<SelleMassSpringImplcitHeptadiagnoalSolver>(massSpringConf);

		auto boneSkinningUpdater = integrator.addSolver<BoneSkinningAnimationDataUpdater>(&bkad);

		SDFCollisionSolver *sdfCollisionSolverPtr = nullptr;
		if (ini.GetBoolean("sdf", "enable")) {
			auto cudaMemoryConverter = integrator.addSolver<CudaMemoryConverter>(Pos_ | Vel_ | LocalIndex_);

			auto sdfCollisionConf = SDFCollisionConfiguration {
					{ini.GetInteger("sdf", "resolutionx"), ini.GetInteger("sdf", "resolutiony"), ini.GetInteger("sdf", "resolutionz")},
					ini.GetReal("sdf", "bbox_extend"),
					ini.GetInteger("sdf", "margin"),
					0.0f,
					ini.GetReal("sdf", "friction"),
					ini.GetReal("sdf", "degeneration_factor"),
					ini.GetBoolean("sdf", "change_hair_root"),
					32 * ini.GetInteger("sdf", "cuda_wrap_size")
			};
			auto sdfCollisionSolver = integrator.addSolver<SDFCollisionSolver>(sdfCollisionConf, boneSkinningUpdater.get());
			auto cudaMemoryInverseConverter = integrator.addSolver<CudaMemoryInverseConverter>(cudaMemoryConverter.get());
		}
		else {
			integrator.addSolver<PositionCommiter>();
		}

		if (ini.GetBoolean("visualize", "hair_enable")) {
			auto hairVplyVisualizer = integrator.addSolver<HairVisualizer>(
					ini.Get("visualize", "hair_folder"),
					ini.Get("visualize", "hair_name_pattern"),
					bkad.getFrameTimeInterval(),
					nullptr
			);
		}

		if (ini.GetBoolean("visualize", "spring_enable")) {
			auto springVplyVisualizer = integrator.addSolver<SelleMassSpringVisualizer>(
					ini.Get("visualize", "spring_folder"),
					ini.Get("visualize", "spring_name_pattern"),
					bkad.getFrameTimeInterval(),
					massSpringSolver.get()
			);
		}

		if (ini.GetBoolean("visualize", "hair_contacts_enable")) {
			auto hairContactsVisualizer = integrator.addSolver<HairContactsAndCollisionImpulseSolverVisualizer>(
					ini.Get("visualize", "hair_contacts_folder"),
					ini.Get("visualize", "hair_contacts_name_pattern"),
					bkad.getFrameTimeInterval(),
					hairContactsSolverPtr,
					nullptr
			);
		}

		if (ini.GetBoolean("visualize", "hair_collisions_enable")) {
			auto hairContactsVisualizer = integrator.addSolver<HairContactsAndCollisionImpulseSolverVisualizer>(
					ini.Get("visualize", "hair_collisions_folder"),
					ini.Get("visualize", "hair_collisions_name_pattern"),
					bkad.getFrameTimeInterval(),
					nullptr,
					hairCollisionSolverPtr
			);
		}

		if (ini.GetBoolean("visualize", "sdf_collisions_enable")) {
			auto sdfCollisionVisualizer = integrator.addSolver<SDFCollisionVisualizer>(
					ini.Get("visualize", "sdf_collisions_folder"),
					ini.Get("visualize", "sdf_collisions_name_pattern"),
					bkad.getFrameTimeInterval(),
					sdfCollisionSolverPtr
			);
		}

		gravitySolver->setMass(&massSpringSolver->getParticleMass());

		float simulationTime = 1.0f / ini.GetReal("integrator", "integration_fps");

		int totalSimulationFrame = static_cast<int>(ini.GetInteger("common", "simulation_frame_count")
		                                            * bkad.getFrameTimeInterval() / simulationTime);

		for (int i = 1; i <= totalSimulationFrame; i += 1) {
			cout << "Simulation Frame " << i << "..." << endl;
			Eigen::Affine3f currentBoneTransform = bkad.getBoneTransform(0, i * simulationTime);
			integrator.simulate(simulationTime, currentBoneTransform);
		}
	} catch (const INIReaderParseError & e) {
		std::cout << e.what() << std::endl;
		throw;
	}
}
