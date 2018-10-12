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

//void validHairContactsCudaSolver() {
//	const auto hair = make_shared<Hair>("/Users/vivi/Developer/Project/HairEngine/Houdini/Resources/Models/Feamle 04 Retop/Hair/Straight-50000-p25.hair");
//
//	Integrator integrator(hair, Eigen::Affine3f::Identity());
//
//	// Copy the particles
//
//	auto cmc = integrator.addSolver<CudaMemoryConverter>(Pos_ | Impulse_ | StrandIndex_);
//	auto smc = integrator.addSolver<CudaSegmentMidpointComputer>();
//	auto impulseSolver = integrator.addSolver<HairContactsImpulseCudaSolver>(smc.get(), 0.001f, 0.002f, 10, 450.0f, 1, 16);
//
//	// Only simulate one frame
//	integrator.simulate(1.0f / 120.0f, Eigen::Affine3f::Identity());
//
//	// Copy the contacts and numContacts back
//	int *contactsCheck = new int[hair->nsegment * impulseSolver->maxContacts];
//	int *numContactsCheck = new int[hair->nsegment];
//
//	CudaUtility::copyFromDeviceToHost(contactsCheck, impulseSolver->contacts, hair->nsegment * impulseSolver->maxContacts);
//	CudaUtility::copyFromDeviceToHost(numContactsCheck, impulseSolver->numContacts, hair->nsegment);
//
//	int totalContacts = 0;
//	for (int sid1 = 0; sid1 < hair->nsegment; ++sid1) {
//
//		if (!(numContactsCheck[sid1] >= 0 && numContactsCheck[sid1] <= impulseSolver->maxContacts)) {
//			printf("Invalid numContacts in sid: %d, with numContacts: %d\n", sid1, numContactsCheck[sid1]);
//			return;
//		}
//
//		for (int i = 0; i < numContactsCheck[sid1]; ++i) {
//			int sid2 = contactsCheck[sid1 * impulseSolver->maxContacts + i];
//			float l = (hair->segments[sid1].midpoint() - hair->segments[sid2].midpoint()).norm();
//			if (l >= impulseSolver->lCreate) {
//				printf("Invalid segment contacts sid1: %d, sid2: %d, with l: %f,while lCreate: %f\n", sid1, sid2, l, impulseSolver->lCreate);
//				return;
//			}
//		}
//
//		totalContacts += numContactsCheck[sid1];
//	}
//
//	printf("Toatal contacts: %d, average contacts: %f\n", totalContacts, static_cast<float>(totalContacts) / hair->nsegment);
//
//	auto psh = impulseSolver->psh;
//	int *hashParStarts = new int[psh->numHash];
//	int *hashParEnds = new int[psh->numHash];
//	int *pids = new int[psh->numHash];
//
//	CudaUtility::copyFromDeviceToHost(hashParStarts, psh->hashParStartsDevice, psh->numHash);
//	CudaUtility::copyFromDeviceToHost(hashParEnds, psh->hashParEndsDevice, psh->numHash);
//	CudaUtility::copyFromDeviceToHost(pids, psh->pidsDevice, psh->numParticle);
//
//	int emptyHash = 0;
//	int totalCell = 0;
//	std::vector<int> particlePerHashDist(35, 0);
//	std::vector<int> cellPerHashDist(35, 0);
//
//	const auto & comp = [](const int3 & i1, const int3 & i2) -> bool {
//		return (i1.x < i2.x) || (i1.x == i2.x && i1.y < i2.y) || (i1.x == i2.x && i1.y == i2.y && i1.z < i2.z);
//	};
//
//	for (int i = 0; i < psh->numHash; ++i) {
//		if (hashParStarts[i] == 0xffffffff) {
//			++emptyHash;
//			continue;
//		}
//
//		std::set<int3, decltype(comp)> cells(comp);
//		for (int t = hashParStarts[i]; t != hashParEnds[i]; ++t) {
//			int sid = pids[t];
//			Eigen::Vector3f midpoint = hair->segments[sid].midpoint();
//			float3 midpoint_ { midpoint.x(), midpoint.y(), midpoint.z() };
//			int3 index3 = make_int3(midpoint_ * psh->dInv);
//			cells.insert(index3);
//		}
//
//		int numCellInHash = cells.size();
//		int numParticleInHash = hashParEnds[i] - hashParStarts[i];
//
//		totalCell += numCellInHash;
//		++particlePerHashDist[std::min(numParticleInHash, 34)];
//		++cellPerHashDist[std::min(numCellInHash, 34)];
//	}
//
//	printf("Total hash: %d, empty: %d, value: %d\n", psh->numHash, emptyHash, psh->numHash - emptyHash);
//	printf("Total segments: %d, total cells: %d\n", psh->numParticle, totalCell);
//	printf("Particel per hash distribution: \n");
//	for (int i = 0; i < 35; ++i) {
//		printf("%d:%d ", i, particlePerHashDist[i]);
//		if (i % 7 == 0)
//			printf("\n");
//	}
//	printf("\n");
//	printf("Cell per hash distribution: \n");
//	for (int i = 0; i < 35; ++i) {
//		printf("%d:%d ", i, cellPerHashDist[i]);
//		if (i % 7 == 0)
//			printf("\n");
//	}
//}

int main(int argc, char **argv) {

	//validHairContactsCudaSolver();

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

		HairContactsImpulseCudaSolver *hairContactsSolverPtr = nullptr;
		CollisionImpulseSolver *hairCollisionSolverPtr = nullptr;

		if (enableHairContacts || enableHairCollisions) {
//			float knnRadius = 0.0f;
//			if (enableHairContacts)
//				knnRadius = std::max(knnRadius, ini.GetReal("haircontacts", "creating_distance"));
//			if (enableHairCollisions)
//				knnRadius = std::max(knnRadius, ini.GetReal("haircollisions", "check_distance"));
//
//			auto segmentKnnSolver = integrator.addSolver<SegmentKNNSolver>(knnRadius);

			if (enableHairContacts) {
				auto cmc = integrator.addSolver<CudaMemoryConverter>(Pos_ | Impulse_ | StrandIndex_);
				auto smc = integrator.addSolver<CudaSegmentMidpointComputer>();
				//auto impulseSolver = integrator.addSolver<HairContactsImpulseCudaSolver>(smc.get(), 0.001f, 0.002f, 10, 450.0f, 1, 16);
				auto hairContactsSolver = integrator.addSolver<HairContactsImpulseCudaSolver>(
						smc.get(),
						ini.GetReal("haircontacts", "creating_distance"),
						ini.GetReal("haircontacts", "breaking_distance"),
						ini.GetInteger("haircontacts", "max_contacts"),
						ini.GetReal("haircontacts", "stiffness")
				);
				auto cmcInv = integrator.addSolver<CudaMemoryInverseConverter>(cmc.get(), Impulse_);

				hairContactsSolverPtr = hairContactsSolver.get();
			}

//			if (enableHairCollisions) {
//				auto hairCollisionSolver = integrator.addSolver<CollisionImpulseSolver>(
//						segmentKnnSolver.get(),
//						ini.GetInteger("haircollisions", "max_collisions"),
//						ini.GetReal("haircollisions", "stiffness"),
//						ini.GetInteger("haircollisions", "max_collisions_force_count")
//				);
//
//				hairCollisionSolverPtr = hairCollisionSolver.get();
//			}
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
//			integrator.addSolver<PositionCommiter>();
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
			auto hairContactsVisualizer = integrator.addSolver<HairContactsImpulseCudaVisualizer>(
					ini.Get("visualize", "hair_contacts_folder"),
					ini.Get("visualize", "hair_contacts_name_pattern"),
					bkad.getFrameTimeInterval(),
					hairContactsSolverPtr
			);
		}

//		if (ini.GetBoolean("visualize", "hair_collisions_enable")) {
//			auto hairContactsVisualizer = integrator.addSolver<HairContactsAndCollisionImpulseSolverVisualizer>(
//					ini.Get("visualize", "hair_collisions_folder"),
//					ini.Get("visualize", "hair_collisions_name_pattern"),
//					bkad.getFrameTimeInterval(),
//					nullptr,
//					hairCollisionSolverPtr
//			);
//		}

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
