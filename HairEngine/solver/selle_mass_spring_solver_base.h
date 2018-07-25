#pragma once

#include <numeric>
#include <functional>

#include "../solver/integration_info.h"
#include "../precompiled/precompiled.h"
#include "../solver/solver.h"
#include "../util/mathutil.h"

namespace HairEngine {
	/**
	 * The base class of the paper Andrew Selle's Mass Spring Model for Hair Simulation.
	 * In this paper, the hair is represented with a group of connected mass spring with some virual 
	 * particles. The base class handle the setup and memory allocation process. The integration is done 
	 * in the derived class.
	 */
	class SelleMassSpringSolverBase: public Solver {
	HairEngine_Public:

		enum {
			PARTICLE_TYPE_INDICATOR_BIT = std::numeric_limits<size_t>::max() - (std::numeric_limits<size_t>::max() >> 1)
		};

		/**
		 * Constructor
		 * 
		 * @param stretchStiffness Stiffness of the stretch spring
		 * @param bendingStiffness Stiffness of the bending spring
		 * @param torsionStiffness Stiffness of the torsion spring
		 * @param damping The damping coefficient
		 * @param maxIntegrationTime The integration time limitation, if the max integration time is larger
		 * than the simulation time, it will use only one iteration
		 * @param enableStrainLimiting Enable strain limiting to protect the inextensibility of the hair
		 * @param colinearMaxDegree We will insert additional virtual particles if two adjacent line segments are "nearly" colinear, 
		 * we treat the two adjacent line segment colinear is the included angle is less than colinearMaxDegree
		 * @param The mass of the single hair strand
		 */
		SelleMassSpringSolverBase(
			float stretchStiffness = 50000.0f, 
			float bendingStiffness = 8000.0f, 
			float torsionStiffness = 8000.0f,
			float damping = 15.0f,
			float maxIntegrationTime = 0.01f,
			bool enableStrainLimiting = true,
			float colinearMaxDegree = 4.0f,
			float mass = 25.0f
		): stretchStiffness(stretchStiffness), bendingStiffness(bendingStiffness), 
			torsionStiffness(torsionStiffness), damping(damping), maxIntegrationTime(maxIntegrationTime),
			enableStrainLimiting(enableStrainLimiting), colinearMaxRad(colinearMaxDegree * 3.141592f / 180.0f), 
			mass(mass) {}

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
			particleIndices = std::allocator<size_t>().allocate(hair.nsegment + nnormal);
			nparticleInStrand = std::allocator<size_t>().allocate(nstrand);
			startIndexForStrand = std::allocator<size_t>().allocate(nstrand);

			// Create virtual particles
			nvirual = 0;
			size_t nindex = 0;

			for (size_t i = 0; i < hair.nstrand; ++i) {
				auto s = hair.strands + i;

				Eigen::Vector3f dirVec = Eigen::Vector3f::Zero();
				Eigen::Vector3f prevD = s->segmentInfo.beginPtr->d();
				size_t nvirtualLocal = 0;

				for (auto seg = s->segmentInfo.beginPtr + 1; seg != s->segmentInfo.endPtr; ++seg) {
					// Add the normal particle of seg->p1
					particleIndices[nindex++] = seg->p1->globalIndex;

					auto d = seg->d();

					if (MathUtility::isColinear(prevD, d, colinearMaxRad)) {
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
							virtualParticles + nvirual,
							currentTransformInverse * virtualParticlePos, // Rest position
							virtualParticlePos, // Position
							Eigen::Vector3f::Zero(), // Velocity
							Eigen::Vector3f::Zero(), // Impluse
							nvirtualLocal, // Local index
							nvirual, // Global index
							s // Strand ptr
						);

						// Add the virtual particle to the indices
						particleIndices[nindex++] = nvirual | PARTICLE_TYPE_INDICATOR_BIT;

						++nvirtualLocal;
						++nvirual;
					}

					// Add the last normal particle to the indices
					particleIndices[nindex++] = (s->particleInfo.endPtr() - 1)->globalIndex;

					nparticleInStrand[i] = nvirtualLocal + s->particleInfo.nparticle;
					startIndexForStrand[i] = (i > 0) ? startIndexForStrand[i - 1] + nparticleInStrand[i] : 0;

					// Next iteration
					prevD = d;
				}
			}

			nparticle = nnormal + nvirual;

			// Allocate the buffer
			HairEngine_AllocatorAllocate(pos1, nparticle);
			HairEngine_AllocatorAllocate(pos2, nparticle);
			HairEngine_AllocatorAllocate(vel1, nparticle);
			HairEngine_AllocatorAllocate(vel2, nparticle);
			HairEngine_AllocatorAllocate(dv1, nparticle);

			// Compute the stiffness and the pmass
			auto prevP = p(0);
			totalLength = 0.0f;
			for (size_t i = 1; i < nparticle; ++i) {
				auto curP = p(i);
				if (curP->strandPtr == prevP->strandPtr)
					totalLength += (curP->pos - prevP->pos).norm();

				// Next iteration
				prevP = curP;
			}

			averageLength = totalLength / nstrand;
			kStretch = stretchStiffness / averageLength;
			kBending = bendingStiffness / averageLength;
			kTorsion = torsionStiffness / averageLength;
			pmass = mass * nstrand / nparticle;
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {
			// Split the time based on the maxIntegrationTime
			float t = maxIntegrationTime / info.t;

			// Make the slices
			std::vector<float> ts;
			ts.push_back(0.0f);
			while (ts.back() + t <= 0.995f) {
				// To avoid float point error, we choose 0.995f
				ts.push_back(ts.back() + t);
			}
			ts.push_back(1.0f);

			// Create the integration info
			auto slicedInfos = info.lerp(ts.begin(), ts.end());

			for (const auto & slicedInfo : slicedInfos)
				_solve(slicedInfo);
		}

		void tearDown() override {
			HairEngine_AllocatorDeallocate(virtualParticles, hairPtr->nsegment);
			HairEngine_AllocatorDeallocate(particleIndices, hairPtr->nsegment + nnormal);

			HairEngine_AllocatorDeallocate(nparticleInStrand, nstrand);
			HairEngine_AllocatorDeallocate(startIndexForStrand, nstrand);

			HairEngine_AllocatorDeallocate(pos1, nparticle);
			HairEngine_AllocatorDeallocate(pos2, nparticle);
			HairEngine_AllocatorDeallocate(vel1, nparticle);
			HairEngine_AllocatorDeallocate(vel2, nparticle);
			HairEngine_AllocatorDeallocate(dv1, nparticle);
		}

	HairEngine_Protected:
		float stretchStiffness;
		float bendingStiffness;
		float torsionStiffness;
		float damping;
		float maxIntegrationTime;
		bool enableStrainLimiting;
		float colinearMaxRad;
		float mass;

		const Hair *hairPtr;

		float totalLength = 0.0f; ///< Total length of the hair
		float averageLength = 0.0f; ///< Average length of the hair strand

		/// Different from stiffness, the real Hooke's stiffness of spring k is defined 
		/// as k = stiffness / averageHairLength, so that the k will change for different 
		/// hair style without changing the stiffness. The reason behind this is that short 
		/// hair needs a larger k for inextensibility.
		float kStretch = 0.0f, kBending = 0.0f, kTorsion = 0.0f;

		float pmass; ///< Particle mass, equals to (mass * nstrand / nparticle)

		size_t nvirual = 0, nnormal = 0, nparticle = 0; ///< Number of normal particles and virtual particles
		Hair::Particle *virtualParticles = nullptr; ///< The virtual particle array
		Hair::Particle *normalParticles = nullptr; ///< The normal particle array
		size_t nstrand; /// Number of strand

		/// The index of all particles (including virtual and normal particles).
		/// We use the highest bit of int to distinguish whether the particle is virtual or not.
		/// A highests bit with 0 to indicate it is a normal particle while 1 indicating it is a virtual particle.
		size_t *particleIndices = nullptr;

		/// An array indicating how many particles in the strand
		size_t *nparticleInStrand = nullptr;
		
		/// The starting index in "particleIndices" for strand
		size_t *startIndexForStrand = nullptr;

		Eigen::Vector3f *pos1 = nullptr, *pos2 = nullptr; ///< Position buffers
		Eigen::Vector3f *dv1 = nullptr, *vel2 = nullptr; ///< Velocity difference buffers
		Eigen::Vector3f *vel1 = nullptr; ///< Velocity buffers

		//TODO: Setup the ParticleToParticle Spring
		

		/* Herlper Function */

		/**
		 * Check whether it is a normal particle based on the index
		 * 
		 * @param index The particle index in the particleIndices
		 * @return True if it is a virual particle
		 */
		inline bool isNormalParticle(size_t index) const {
			return index & PARTICLE_TYPE_INDICATOR_BIT == 0;
		}

		/**
		 * Check whether it is a virual particle based on the index
		 * 
		 * @param index The particle index in the particleIndices
		 * @return True if it is a virtual particle
		 */
		inline bool isVirtualParticle(size_t index) const { return !isNormalParticle(index); }

		/**
		 * Get the particle pointer based on the index
		 * 
		 * @param index The index of the particle in the particleIndices
		 * @return The particle pointer
		 */
		inline Hair::Particle::Ptr p(size_t index) const {
			return isNormalParticle(index) ? (normalParticles + index) : (virtualParticles + (index - PARTICLE_TYPE_INDICATOR_BIT));
		}

		/**
		 * Helper function for doing some operations to all particles. 
		 * It should guarantee that the order for modification will not change the result. 
		 * 
		 * @param parallel Enable parallism for mapping. If it is false, we will do it sequentially.
		 * @param modifier A callable function like object, we will pass the pointer and its index in the particleIndices to 
		 * the modifier.
		 */
		void mapParticle(bool parallel, const std::function<void(Hair::Particle::Ptr, size_t)> & modifier) {
			if (parallel) {
				// Not implemented
				assert(false);
			}
			else {
				// Sequential
				for (size_t i = 0; i < nparticle; ++i)
					modifier(p(i), i);
			}
		}

		/**
		 * Implementation of the integration. Subclass should provide 
		 * its own method for integration. Subclass fetch the position from 
		 * "pos" and velocity from "vel", update the "dv" for velocity difference.
		 * 
		 * @param pos The position buffer (Read)
		 * @param vel The velocity buffer (Read)
		 * @param dv The velocity difference buffer (Write)
		 * @param t The integration time
		 */
		virtual void integrate(Eigen::Vector3f *pos, Eigen::Vector3f *vel, Eigen::Vector3f *dv, float t) = 0;

		/**
		 * Solve with a small time slice integration. Use a variant of 
		 * Runge Kutta integration.
		 * 
		 * @param info Sliced intergration info
		 */
		virtual void _solve(const IntegrationInfo & info) {

			float t_2 = info.t / 2.0f;

			// Store the position and velocity into buffer
			mapParticle(false, [this, &](Hair::Particle::Ptr par, size_t i) {
				pos1[i] = par->pos;
				vel1[i] = par->vel;
			});

			// First integration
			integrate(pos1, vel1, dv1, t_2);

			// Strain limiting
			// TODO: Strain limiting

			// Compute middle properties
			mapParticle(false, [this](Hair::Particle::Ptr par, size_t i) {
				vel2[i] = vel1[i] + dv1[i]; // vel2 is stored the middle velocity
				pos2[i] = pos1[i] + vel2[i] * t_2; // pos2 is stored the middle position
			});

			// Second integration
			integrate(pos2, vel2, dv1, t_2);

			// Update the final velocity
			mapParticle(false, [this](Hair::Particle::Ptr par, size_t i) {
				par->vel += 2.0f * dv1;
			});
		}
	};
}