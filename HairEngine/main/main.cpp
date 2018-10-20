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
#include "../solver/hair_contacts_impulse_cuda_solver.h"
#include "../solver/collision_impulse_cuda_solver.h"

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
		float number;
		is >> number;
		if (is)
			xy.emplace_back(number);
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

		bool useStaticHairRoot = ini.GetBoolean("hair", "static_hair_root");
		Eigen::Affine3f initialBoneTransform = useStaticHairRoot ? Eigen::Affine3f::Identity() : bkad.getRestBoneTransform(0);

		// Generate an empty hair
		int resampleRate = ini.GetBoolean("hair", "enable_resampling") ? ini.GetInteger("hair", "resample_rate") : 1;
		const auto hair = make_shared<Hair>(Hair(ini.Get("hair", "path"),
		                                         initialBoneTransform.inverse(Eigen::Affine)).resample(resampleRate));

		float visualizeTime = ini.GetBoolean("visualize", "timming_override") ? 0.0f : bkad.getFrameTimeInterval();

		cout << "Creating integrator..." << endl;
		Integrator integrator(hair, initialBoneTransform);

		Eigen::Vector3f gravity = Eigen::Vector3f(
				ini.GetReal("common", "gravityx"),
				ini.GetReal("common", "gravityy"),
				ini.GetReal("common", "gravityz")
		);
		auto gravitySolver = integrator.addSolver<FixedAccelerationApplier>(true, gravity);

		HairContactsImpulseCudaSolver *hairContactsSolverPtr = nullptr;
		CollisionImpulseCudaSolver *collisionImpulseSolverPtr = nullptr;

		// Add hair contacts and collisions solver
		{
			auto enableHairContacts = ini.GetBoolean("haircontacts", "enable");
			auto enableHairCollisions = ini.GetBoolean("haircollisions", "enable");

			CudaSegmentMidpointComputer *smcPtr = nullptr;
			CudaMemoryConverter *cmcPtr = nullptr;

			// Add additional solver for hair contacts and impulse collision
			if (enableHairContacts || enableHairCollisions) {
				int copyOptions = 0;
				if (enableHairContacts)
					copyOptions |= Pos_ | Impulse_ | StrandIndex_;
				if (enableHairCollisions)
					copyOptions |= Pos_ | Vel_ | Impulse_ | StrandIndex_;

				cmcPtr = integrator.addSolver<CudaMemoryConverter>(copyOptions).get();

				smcPtr = integrator.addSolver<CudaSegmentMidpointComputer>().get();
			}

			if (enableHairContacts)
				hairContactsSolverPtr = integrator.addSolver<HairContactsImpulseCudaSolver>(
						smcPtr,
						ini.GetReal("haircontacts", "creating_distance"),
						ini.GetReal("haircontacts", "breaking_distance"),
						ini.GetInteger("haircontacts", "max_contacts"),
						ini.GetReal("haircontacts", "stiffness"),
						ini.GetReal("haircontacts", "resolution"),
						ini.GetInteger("haircontacts", "cuda_wrap_size")
				).get();

			if (enableHairCollisions)
				collisionImpulseSolverPtr = integrator.addSolver<CollisionImpulseCudaSolver>(
						smcPtr,
						ini.GetInteger("haircollisions", "max_collisions"),
						ini.GetInteger("haircollisions", "max_collisions_force_count"),
						ini.GetReal("haircollisions", "stiffness"),
						ini.GetReal("haircollisions", "resolution"),
						ini.GetInteger("haircollisions", "cuda_wrap_size")
				).get();

			if (enableHairContacts || enableHairCollisions)
				integrator.addSolver<CudaMemoryInverseConverter>(cmcPtr, Impulse_);
		}

		auto massSpringConf = SelleMassSpringSolverBase::Configuration(
				ini.GetReal("massspring", "stretch_stiffness"),
				ini.GetReal("massspring", "bending_stiffness"),
				ini.GetReal("massspring", "torsion_stiffness"),
				ini.GetReal("massspring", "altitude_stiffness"),
				ini.GetReal("massspring", "damping"),
				getVaryingFloat(ini.Get("massspring", "rigidness")),
				ini.GetBoolean("massspring", "enable_strain_limiting") ? ini.GetReal("massspring", "strain_limiting_tolerance") : 0.0f,
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

			sdfCollisionSolverPtr = sdfCollisionSolver.get();
		}
		else {
			integrator.addSolver<PositionCommiter>();
		}

		if (ini.GetBoolean("visualize", "hair_enable")) {
			auto hairVplyVisualizer = integrator.addSolver<HairVisualizer>(
					ini.Get("visualize", "hair_folder"),
					ini.Get("visualize", "hair_name_pattern"),
					visualizeTime,
					nullptr
			);
		}

		if (ini.GetBoolean("visualize", "spring_enable")) {
			auto springVplyVisualizer = integrator.addSolver<SelleMassSpringVisualizer>(
					ini.Get("visualize", "spring_folder"),
					ini.Get("visualize", "spring_name_pattern"),
					visualizeTime,
					massSpringSolver.get()
			);
		}

		if (ini.GetBoolean("visualize", "hair_contacts_enable")) {
			auto hairContactsVisualizer = integrator.addSolver<HairContactsImpulseCudaVisualizer>(
					ini.Get("visualize", "hair_contacts_folder"),
					ini.Get("visualize", "hair_contacts_name_pattern"),
					visualizeTime,
					hairContactsSolverPtr
			);
		}

		if (ini.GetBoolean("visualize", "hair_collisions_enable")) {
			auto hairContactsVisualizer = integrator.addSolver<CollisionImpulseCudaVisualizer>(
					ini.Get("visualize", "hair_collisions_folder"),
					ini.Get("visualize", "hair_collisions_name_pattern"),
					visualizeTime,
					collisionImpulseSolverPtr
			);
		}

		if (ini.GetBoolean("visualize", "sdf_collisions_enable")) {
			auto sdfCollisionVisualizer = integrator.addSolver<SDFCollisionVisualizer>(
					ini.Get("visualize", "sdf_collisions_folder"),
					ini.Get("visualize", "sdf_collisions_name_pattern"),
					visualizeTime,
					sdfCollisionSolverPtr
			);
		}

		gravitySolver->setMass(&massSpringSolver->getParticleMass());

		float simulationTime = 1.0f / ini.GetReal("integrator", "integration_fps");

		int totalSimulationFrame = static_cast<int>(ini.GetInteger("common", "simulation_frame_count")
		                                            * bkad.getFrameTimeInterval() / simulationTime);

		for (int i = 1; i <= totalSimulationFrame; i += 1) {
			cout << "Simulation Frame " << i << "..." << endl;
			Eigen::Affine3f currentBoneTransform = useStaticHairRoot ? Eigen::Affine3f::Identity() : bkad.getBoneTransform(0, i * simulationTime);
			integrator.simulate(simulationTime, currentBoneTransform);
		}
	} catch (const INIReaderParseError & e) {
		std::cout << e.what() << std::endl;
		throw;
	}
}
