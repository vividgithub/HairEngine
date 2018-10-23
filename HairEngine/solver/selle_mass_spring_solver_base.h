#pragma once

#include <numeric>
#include <functional>
#include <chrono>

#include "../solver/integration_info.h"
#include "../precompiled/precompiled.h"
#include "../solver/solver.h"
#include "hair_visualizer.h"
#include "../util/mathutil.h"
#include "../util/parallutil.h"
#include "selle_mass_spring_conf.h"

namespace HairEngine {

	class SelleMassSpringVisualizer;

	/**
	 * The base class of the paper Andrew Selle's Mass Spring Model for Hair Simulation.
	 * In this paper, the hair is represented with a group of connected mass spring with some virual 
	 * particles. The base class handle the setup and memory allocation process. The integration is done 
	 * in the derived class.
	 */
	class SelleMassSpringSolverBase: public Solver, public HairVisualizerVirtualParticleVisualizationInterface {

		friend class SelleMassSpringVisualizer;

	HairEngine_Public:

		static constexpr const int PARTICLE_TYPE_INDICATOR_BIT = std::numeric_limits<int>::min();

		/**
		 * Normal spring definition. i1 and i2 indicates the index in the particleIndices for the particle to 
		 * connect. Float k indicates the real stiffness, and l0 is the rest length of the spring.
		 */
		struct Spring {
			int i1, i2;
			float k;
			float l0;
			int32_t typeID; // 0 for Stretch, 1 for Bending, 2 for Torsion

			Spring(int i1, int i2, float k, float l0, int32_t typeID):
				i1(i1), i2(i2), k(k) , l0(l0), typeID(typeID) {}
		};

		using Configuration = SelleMassSpringConfiguration;

		/**
		 * Addtional particle property for simulation
		 */
		struct ParticleProperty {
			float rigidness; ///< The rigidness is assigned to each particle by varying floats
		};

		/**
		 * Constructor
		 * 
		 * @param Configuration The configuration for initialization
		 */
		SelleMassSpringSolverBase( const Configuration & conf): 
			stretchStiffness(conf.stretchStiffness),
			bendingStiffness(conf.bendingStiffness),
			torsionStiffness(conf.torsionStiffness),
			altitudeStiffness(conf.altitudeStiffness),
			damping(conf.damping),
			rigidness(conf.rigidness),
			strainLimitingLengthTolerance(conf.strainLimitingLengthTolerance),
			colinearMaxRad(conf.colinearMaxDegree * 3.141592f / 180.0f),
			mass(conf.mass),
			maxIntegrationTime(conf.maxIntegrationTime) {}

		void setup(const Hair& hair, const Eigen::Affine3f & currentTransform) override {
			// Inverse the transform
			Eigen::Affine3f currentTransformInverse = currentTransform.inverse(Eigen::TransformTraits::Affine);

			// Set up the known variables
			nnormal = hair.nparticle;
			normalParticles = hair.particles;
			nstrand = hair.nstrand;
			hairPtr = &hair;

			// Allocate the space 

			// The virtual particle count is less than the number of segments
			auto particleAllocator = std::allocator<Hair::Particle>();

			virtualParticles = particleAllocator.allocate(hair.nsegment);
			particleIndices = std::allocator<int>().allocate(hair.nsegment + nnormal);
			nparticleInStrand = std::allocator<int>().allocate(nstrand);
			particleStartIndexForStrand = std::allocator<int>().allocate(nstrand);

			// Create virtual particles
			nvirtual = 0;
			int nindex = 0;

			for (int i = 0; i < hair.nstrand; ++i) {
				auto s = hair.strands + i;

				Eigen::Vector3f dirVec = Eigen::Vector3f::Zero();
				Eigen::Vector3f prevD = s->segmentInfo.beginPtr->d();
				int nvirtualLocal = 0;

				for (auto seg = s->segmentInfo.beginPtr; seg != s->segmentInfo.endPtr; ++seg) {
					// Add the normal particle of seg->p1
					particleIndices[nindex++] = seg->p1->globalIndex;

					auto d = seg->d();

					if (false && MathUtility::isColinear(prevD, d, colinearMaxRad)) {
						// Random assign dirVec to a normalized vector until they are not so "colinear"
						if (dirVec == Eigen::Vector3f::Zero()) {
							do {
								dirVec = Eigen::Vector3f::Random();
							} while (MathUtility::isColinear(dirVec, d));
						}

						Eigen::Vector3f normalVec = dirVec.cross(d).normalized() * 0.8660254037844386f * d.norm();
						Eigen::Vector3f virtualParticlePos = seg->p1->pos + (0.5f * d) + normalVec;

						// Create the virual particle
						particleAllocator.construct(
							virtualParticles + nvirtual,
							currentTransformInverse * virtualParticlePos, // Rest position
							virtualParticlePos, // Position
							Eigen::Vector3f::Zero(), // Velocity
							Eigen::Vector3f::Zero(), // Impluse
							nvirtualLocal, // Local index
							nvirtual, // Global index
							s->index // Strand ptr
						);

						// Add the virtual particle to the indices
						particleIndices[nindex++] = nvirtual | PARTICLE_TYPE_INDICATOR_BIT;

						++nvirtualLocal;
						++nvirtual;

						// Reassign the dirVec to allow rotation
						dirVec = normalVec;
					}

					// Next iteration
					prevD = d;
				}

				// Add the last normal particle to the indices
				particleIndices[nindex++] = (s->particleInfo.endPtr - 1)->globalIndex;

				nparticleInStrand[i] = nvirtualLocal + s->particleInfo.nparticle;
				particleStartIndexForStrand[i] = (i > 0) ? particleStartIndexForStrand[i - 1] + nparticleInStrand[i - 1] : 0;
			}

			nparticle = nnormal + nvirtual;

			// Allocate buffer space
			pos1 = new Eigen::Vector3f[nparticle];
			pos2 = new Eigen::Vector3f[nparticle];
			vel1 = new Eigen::Vector3f[nparticle];
			vel2 = new Eigen::Vector3f[nparticle];
			vel3 = new Eigen::Vector3f[nparticle];

			// Compute the stiffness and the pmass
			auto prevP = p(0);
			totalLength = 0.0f;
			for (int i = 1; i < nparticle; ++i) {
				auto curP = p(i);
				if (curP->strandIndex == prevP->strandIndex)
					totalLength += (curP->pos - prevP->pos).norm();

				// Next iteration
				prevP = curP;
			}

			averageLength = totalLength / nstrand;
			kStretch = stretchStiffness / averageLength;
			kBending = bendingStiffness / averageLength;
			kTorsion = torsionStiffness / averageLength;
			kAltitude = altitudeStiffness / averageLength;
			pmass = mass * nstrand / nparticle;

			// Setup the spring
			HairEngine_AllocatorAllocate(springs, nparticle * 3);
			HairEngine_AllocatorAllocate(nspringInStrand, nstrand);
			HairEngine_AllocatorAllocate(springStartIndexForStrand, nstrand);

			nspring = 0;
			for (int si = 0; si < nstrand; ++si) {

				nspringInStrand[si] = nspring;

				for (int i = particleStartIndexForStrand[si]; i < particleStartIndexForStrand[si] + nparticleInStrand[si]; ++i) {
					auto par = p(i);
					Hair::Particle::Ptr par1 = (i + 1 < nparticle) ? p(i + 1) : nullptr;
					Hair::Particle::Ptr par2 = (i + 2 < nparticle) ? p(i + 2) : nullptr;
					Hair::Particle::Ptr par3 = (i + 3 < nparticle) ? p(i + 3) : nullptr;

					// Create stretch, bending and torsion spring
					if (par1 && par1->strandIndex == par->strandIndex)
						std::allocator<Spring>().construct(springs + (nspring++), i, i + 1, kStretch, (par1->restPos - par->restPos).norm(), 0);
					if (par2 && par2->strandIndex == par->strandIndex)
						std::allocator<Spring>().construct(springs + (nspring++), i, i + 2, kBending, (par2->restPos - par->restPos).norm(), 1);
					if (par3 && par3->strandIndex == par->strandIndex)
						std::allocator<Spring>().construct(springs + (nspring++), i, i + 3, kTorsion, (par3->restPos - par->restPos).norm(), 2);
				}

				nspringInStrand[si] = nspring - nspringInStrand[si];
				springStartIndexForStrand[si] = (si > 0) ? springStartIndexForStrand[si - 1] + nspringInStrand[si - 1] : 0;
			}

			// Setup the particle properties
			HairEngine_AllocatorAllocate(particleProps, nparticle);
			mapStrand(true, [this](int si){
				auto parBegin = particleStartIndexForStrand[si];
				auto parEnd = parBegin + nparticleInStrand[si];

				// Get the total length of the strand
				float strandLength = 0.0f;
				for (int i = parBegin + 1; i != parEnd; ++i)
					strandLength += (p(i)->restPos - p(i - 1)->restPos).norm();

				float currentLength = 0.0f;
				for (int i = parBegin; i != parEnd; ++i) {
					float rl = currentLength / strandLength; // Relative length
					particleProps[i].rigidness = rigidness(rl);

					if (i != parEnd - 1)
						currentLength += (p(i + 1)->restPos - p(i)->restPos).norm();
				}
			});
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {

			// Copy to position buffer
			mapParticle(true, [this, &info](Hair::Particle::Ptr par, int i) {
				pos1[i] = par->pos;
				vel1[i] = par->vel;
			});

			// Setup the split time interval
			std::vector<float> timeIntervals = { 0.0f };
			float tinc = (maxIntegrationTime <= 0.0f) ? 1.0f : maxIntegrationTime / info.t;
			while (timeIntervals.back() + tinc <= 1.0f) {
				timeIntervals.push_back(timeIntervals.back() + tinc);
			}
			if (timeIntervals.back() <= 0.995f)
				timeIntervals.push_back(1.0f);

			auto splittedInfos = info.lerp(timeIntervals.cbegin(), timeIntervals.cend());

			for (const auto & splittedInfo : splittedInfos) {

				integrate(pos1, vel1, vel2, splittedInfo);

				ParallismUtility::parallelFor(0, nparticle, [this, &splittedInfo] (int i) {
					pos2[i] = pos1[i] + splittedInfo.t * vel2[i];
				});

				// Enable strain limiting
				if (strainLimitingLengthTolerance > 1.0f) {
					mapStrand(true, [this] (int si) {
						auto parBegin = particleStartIndexForStrand[si];
						auto parEnd = parBegin + nparticleInStrand[si];

						auto pi = parBegin;
						Eigen::Vector3f piPrevPos = pos2[parBegin]; // The previous position before modification

						for (int i = parBegin + 1; i != parEnd; ++i) {
							float ltol = (p(i)->restPos - p(pi)->restPos).norm() * strainLimitingLengthTolerance;

							Eigen::Vector3f d = pos2[i] - piPrevPos;
							float l = d.norm();
							d /= l;

							bool shouldModifyPi = isNormalParticle(i);

							// Modify before the pos2 changes
							if (shouldModifyPi)
								piPrevPos = pos2[i];

							if (l > ltol)
								pos2[i] = pos2[pi] + d * ltol;

							if (shouldModifyPi)
								pi = i;
						}
					});
				}

				// Rigidness: interpolate the velocity
				auto transformFromPrev = splittedInfo.tr * splittedInfo.ptr.inverse(Eigen::Affine);
				ParallismUtility::parallelFor(0, nparticle, [this, &splittedInfo, &transformFromPrev] (int i) {
					pos2[i] = MathUtility::lerp(pos2[i], transformFromPrev * pos1[i], particleProps[i].rigidness);
				});

				std::swap(vel1, vel2);
				std::swap(pos1, pos2);
			}

			// Copy out the result
			mapParticle(true, [this, &info](Hair::Particle::Ptr par, int i) {

				par->vel = (pos1[i] - par->pos) / info.t;
				if (isVirtualParticle(i))
					par->pos = pos1[i];

			});

			//float t_2 = info.t / 2.0f;

			//// Store the position and velocity into buffer
			//mapParticle(false, [this](Hair::Particle::Ptr par, int i) {
			//	pos1[i] = par->pos;
			//	vel1[i] = par->vel;
			//});

			//// First integration
			//integrate(pos1, vel1, vel2, t_2);

			//// Strain limiting
			//// TODO: Strain limiting

			//// Compute middle properties
			//mapParticle(false, [this, t_2](Hair::Particle::Ptr par, int i) {
			//	pos2[i] = pos1[i] + vel2[i] * t_2; // pos2 is stored the middle position
			//});

			//// Second integration
			//integrate(pos2, vel2, vel3, t_2);

			//// Update the final velocity
			//mapParticle(false, [this](Hair::Particle::Ptr par, int i) {
			//	par->vel += 2.0f * (vel3[i] - vel2[i]);
			//});
		}

		void tearDown() override {
			HairEngine_AllocatorDeallocate(virtualParticles, hairPtr->nsegment);
			HairEngine_AllocatorDeallocate(particleIndices, hairPtr->nsegment + nnormal);

			HairEngine_AllocatorDeallocate(nparticleInStrand, nstrand);
			HairEngine_AllocatorDeallocate(particleStartIndexForStrand, nstrand);

			HairEngine_AllocatorDeallocate(springs, nparticle * 3);
			HairEngine_AllocatorDeallocate(nspringInStrand, nstrand);
			HairEngine_AllocatorDeallocate(springStartIndexForStrand, nstrand);

			HairEngine_SafeDeleteArray(pos1);
			HairEngine_SafeDeleteArray(pos2);
			HairEngine_SafeDeleteArray(vel1);
			HairEngine_SafeDeleteArray(vel2);
			HairEngine_SafeDeleteArray(vel3);

			HairEngine_AllocatorDeallocate(particleProps, nparticle);
		}

		/* HairVisualizerVirtualParticleVisualizationInterface Interface */
		const Hair::Particle& getVirtualParticle(int index) const override {
			return virtualParticles[index];
		}

		int virtualParticleSize() const override {
			return nvirtual;
		}

		const float & getParticleMass() const {
			return pmass;
		}

		const double & getIntegrationTime() const {
			return integrationTime;
		}

		int getParticleCount() const {
			return nparticle;
		}

		int getStrandCount() const {
			return nstrand;
		}

	HairEngine_Protected:
		float stretchStiffness;
		float bendingStiffness;
		float torsionStiffness;
		float altitudeStiffness;
		float damping;
		float strainLimitingLengthTolerance;
		float colinearMaxRad;
		float mass;
		float maxIntegrationTime;
		VaryingFloat rigidness;

		const Hair *hairPtr;

		float totalLength = 0.0f; ///< Total length of the hair
		float averageLength = 0.0f; ///< Average length of the hair strand

		/// Different from stiffness, the real Hooke's stiffness of spring k is defined 
		/// as k = stiffness / averageHairLength, so that the k will change for different 
		/// hair style without changing the stiffness. The reason behind this is that short 
		/// hair needs a larger k for inextensibility.
		float kStretch = 0.0f, kBending = 0.0f, kTorsion = 0.0f, kAltitude = 0.0f;

		float pmass; ///< Particle mass, equals to (mass * nstrand / nparticle)

		int nvirtual = 0, nnormal = 0, nparticle = 0; ///< Number of normal particles and virtual particles
		Hair::Particle *virtualParticles = nullptr; ///< The virtual particle array
		Hair::Particle *normalParticles = nullptr; ///< The normal particle array
		int nstrand; /// Number of strand

		/// The index of all particles (including virtual and normal particles).
		/// We use the highest bit of int to distinguish whether the particle is virtual or not.
		/// A highests bit with 0 to indicate it is a normal particle while 1 indicating it is a virtual particle.
		int *particleIndices = nullptr;

		/// An array indicating how many particles in the strand
		int *nparticleInStrand = nullptr;
		
		/// The starting index in "particleIndices" for strand
		int *particleStartIndexForStrand = nullptr;

		Eigen::Vector3f *pos1 = nullptr, *pos2 = nullptr; ///< Position buffers
		Eigen::Vector3f *vel1 = nullptr, *vel3 = nullptr, *vel2 = nullptr; ///< Velocity difference buffers

		Spring *springs; ///< The spring array
		int nspring; ///< The size of spring array
		int *nspringInStrand = nullptr; ///< Number of strand in the strand i
		int *springStartIndexForStrand = nullptr; ///< The start index in the "springs" array for the strand i

		ParticleProperty *particleProps; ///< The addtional property array for particles

		double integrationTime = 0.0f;

		/* Herlper Function */

		/**
		 * Check whether it is a normal particle based on the index
		 * 
		 * @param index The particle index in the particleIndices
		 * @return True if it is a virual particle
		 */
		inline bool isNormalParticle(int index) const {
			return particleIndices[index] >= 0;
		}

		/**
		 * Check whether it is a virual particle based on the index
		 * 
		 * @param index The particle index in the particleIndices
		 * @return True if it is a virtual particle
		 */
		inline bool isVirtualParticle(int index) const { return !isNormalParticle(index); }

		/**
		 * Get the particle pointer based on the index
		 * 
		 * @param index The index of the particle in the squence of normal particles
		 * @return The particle pointer
		 */
		inline Hair::Particle::Ptr p(int index) const {
			index = particleIndices[index];
			return index >= 0 ? (normalParticles + index) : (virtualParticles + (index - PARTICLE_TYPE_INDICATOR_BIT));
		}

		/**
		 * Helper function for doing some operations to all particles. 
		 * It should guarantee that the order for modification will not change the result. 
		 * 
		 * @param parallel Enable parallism for mapping. If it is false, we will do it sequentially.
		 * @param mapper A callable function like object, we will pass the pointer and its index in the particleIndices to 
		 * the modifier.
		 */
		void mapParticle(bool parallel, const std::function<void(Hair::Particle::Ptr, int)> & mapper) {
			const auto & block = [this, &mapper](int i) { mapper(p(i), i); };
			ParallismUtility::conditionalParallelFor(parallel, 0, static_cast<int>(nparticle), block);
		}

		/**
		 * Helper function for doing some operations to all strands.
		 * It should guarantee that the order of the modification will not change the result 
		 * 
		 * @param parallel Enable parallism for mapping. Ohterwise we will do it sequentially
		 * @param mapper A callbale function which accepts a strand index, do some stuff with the index
		 */
		void mapStrand(bool parallel, const std::function<void(int)> & mapper) {
			ParallismUtility::conditionalParallelFor(parallel, 0, static_cast<int>(nstrand), mapper);
		}

		/**
		 * Implementation of the integration. Subclass should provide 
		 * its own method for integration. Subclass fetch the position from 
		 * "pos" and velocity from "vel", update the "dv" for velocity difference.
		 * 
		 * @param pos The position buffer (Read)
		 * @param vel The velocity buffer (Read)
		 * @param outVel The output velocity buffer, it should be different from the vel (Write)
		 * @param info The integration info
		 */
		virtual void integrate(Eigen::Vector3f *pos, Eigen::Vector3f *vel, Eigen::Vector3f *outVel, const IntegrationInfo & info) = 0;
	};
}