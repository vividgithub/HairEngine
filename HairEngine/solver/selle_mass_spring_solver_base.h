#pragma once

#include <numeric>
#include <functional>
#include <chrono>
#include <array>

#include "../solver/integration_info.h"
#include "../precompiled/precompiled.h"
#include "../solver/solver.h"
#include "hair_visualizer.h"
#include "../util/mathutil.h"
#include "../util/parallutil.h"

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

		/**
		 * Altitude spring definition, an altitude spring is represented as a tetrahedra with 4 adajcent particles.
		 * Altitude spring is used for fixing the position degeneration when we arbitarily change the position of the particles 
		 */
		struct AltitudeSpring {
			int i1, i2, i3, i4;
			float k;

			/*
			* A temporary storage to store some caculation variables. In a tetrahedral, we have 3
			* edge/edge springs and 4 point/face spring, so that we use a array of size 7(3 + 4) to store them.
			* The order is:
			*     0. Edge/Edge: {p1, p2} --> {p3, p4}
			*     1. Edge/Edge: {p1, p3} --> {p2, p4}
			*     2. Edge/Edge: {p1, p4} --> {p2, p3}
			*     3. Point/Face: p1 --> {p2, p3, p4}
			*     4. Point/Face: p2 --> {p1, p3, p4}
			*     5. Point/Face: p3 --> {p1, p2, p4}
			*     6. Point/Face: p4 --> {p1, p2, p3}
			* The l0s variables store the rest length of each tetrahedral elements, normals is used for storing the
			* current normals, and vs means a point/point vector which points from p1 to another edge/face vertex
			* (since we want the i, and d are pointed from p1), for example, for a Point/Face p4 --> {p1, p2, p3},
			* the v is p1 --> p4.
			*/
			std::array<float, 7> l0s; ///< The rest length of each tetrahedral point/face or edge/edge
			
			AltitudeSpring(int i1, int i2, int i3, int i4, float k, const std::array<float, 7> & l0s):
				i1(i1), i2(i2), i3(i3), i4(i4), k(k), l0s(l0s) {}
		};

		/**
		 * Constructor configuration to initialize a SelleMassSpringSolverBase
		 */
		struct Configuration {
			float stretchStiffness; ///< Stiffness of the stretch spring
			float bendingStiffness; ///< Stiffness of the bending spring
			float torsionStiffness; ///< Stiffness of the torsion spring
			float altitudeStiffness; ///< Stiffness of the altitude spring
			float damping; ///< The damping coefficient
			bool enableStrainLimiting; ///< Enable strain limiting to protect the inextensibility of the hair
			float colinearMaxDegree; ///< We will insert additional virtual particles if two adjacent line segments are "nearly" colinear, we treat the two adjacent line segment colinear is the included angle is less than colinearMaxDegree
			float mass; ///< The mass of the single hair strand

			/**
			 * Constructor
			 */
			Configuration(
				float stretchStiffness,
				float bendingStiffness,
				float torsionStiffness,
				float altitudeStiffness,
				float damping,
				bool enableStrainLimiting,
				float colinearMaxDegree,
				float mass
			): stretchStiffness(stretchStiffness), bendingStiffness(bendingStiffness), 
			torsionStiffness(torsionStiffness), altitudeStiffness(altitudeStiffness), damping(damping),
			enableStrainLimiting(enableStrainLimiting), colinearMaxDegree(colinearMaxDegree), mass(mass) {}
		};

		/**
		 * Constructor
		 * 
		 * @param Configuration The configuration for initialization
		 */
		SelleMassSpringSolverBase( const Configuration & conf): 
			stretchStiffness(conf.stretchStiffness), bendingStiffness(conf.bendingStiffness), 
			torsionStiffness(conf.torsionStiffness), altitudeStiffness(conf.altitudeStiffness), damping(conf.damping),
			enableStrainLimiting(conf.enableStrainLimiting), colinearMaxRad(conf.colinearMaxDegree * 3.141592f / 180.0f), 
			mass(conf.mass) {}

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

			//Setup the altitude springs
			HairEngine_AllocatorAllocate(altitudeSprings, nparticle); 
			HairEngine_AllocatorAllocate(naltitudeInStrand, nstrand);
			HairEngine_AllocatorAllocate(altitudeStartIndexForStrand, nstrand);

			naltitude = 0;
			for (int si = 0; si < nstrand; ++si) {

				naltitudeInStrand[si] = 0;

				for (int i = particleStartIndexForStrand[si]; i < particleStartIndexForStrand[si] + nparticleInStrand[si] - 3; ++i) {
					auto p1 = p(i);
					Hair::Particle::Ptr p2 = p(i + 1);
					Hair::Particle::Ptr p3 = p(i + 2);
					Hair::Particle::Ptr p4 = p(i + 3);

					// Create altitude springs
					Eigen::Vector3f vs[7], normals[7];

					std::array<float, 7> l0s;

					Eigen::Vector3f
						d12 = p2->restPos - p1->restPos,
						d13 = p3->restPos - p1->restPos,
						d14 = p4->restPos - p1->restPos,
						d23 = p3->restPos - p2->restPos,
						d24 = p4->restPos - p2->restPos,
						d34 = p4->restPos - p3->restPos;

					vs[0] = d13;
					vs[1] = d12;
					vs[2] = d12;

					vs[3] = d12;
					vs[4] = d12;
					vs[5] = d13;
					vs[6] = d14;

					normals[0] = d12.cross(d34);
					normals[1] = d13.cross(d24);
					normals[2] = d14.cross(d23);

					normals[3] = d23.cross(d24);
					normals[4] = d13.cross(d14);
					normals[5] = d12.cross(d14);
					normals[6] = d12.cross(d23);

					for (int i = 0; i < 7; ++i) {
						normals[i].normalize();
						l0s[i] = MathUtility::project(vs[i], normals[i]).norm();
					}

					std::allocator<AltitudeSpring>().construct(altitudeSprings + (naltitude++), i, i + 1, i + 2, i + 3, kAltitude, l0s);
					++naltitudeInStrand[si];
				}

				altitudeStartIndexForStrand[si] = (si > 0) ? altitudeStartIndexForStrand[si - 1] + naltitudeInStrand[si - 1] : 0;
			}
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {

			mapParticle(true, [this, &info](Hair::Particle::Ptr par, int i) {
				pos1[i] = par->pos;
				vel1[i] = par->vel;
			});

			const auto startIntegration = std::chrono::high_resolution_clock::now();
			integrate(pos1, vel1, vel2, info);
			const auto endIntegration = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double> diff = endIntegration - startIntegration;
			integrationTime = diff.count();

			mapParticle(true, [this, &info](Hair::Particle::Ptr par, int i) {
				par->vel = vel2[i];

				// Only commit the position of the virtual particles
				if (isVirtualParticle(i))
					par->pos += info.t * par->vel;
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

			HairEngine_AllocatorDeallocate(altitudeSprings, nparticle);
			HairEngine_AllocatorDeallocate(naltitudeInStrand, nstrand);
			HairEngine_AllocatorDeallocate(altitudeStartIndexForStrand, nstrand);

			HairEngine_SafeDeleteArray(pos1);
			HairEngine_SafeDeleteArray(pos2);
			HairEngine_SafeDeleteArray(vel1);
			HairEngine_SafeDeleteArray(vel2);
			HairEngine_SafeDeleteArray(vel3);
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

		AltitudeSpring *altitudeSprings; ///< The altitude spring arrays
		int naltitude; ///< The size of altitude spring
		int *naltitudeInStrand = nullptr; ///< Number of altitude spring in the strand i
		int *altitudeStartIndexForStrand = nullptr; ///< The start index of altitude spring in the "altitudeSprings" array in the strand i

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

		/**
		 * Get the altitude spring information based on current state. The selectedIndex indicates which point/face or edge/edge 
		 * components are selected for the altitude spring. The "d" indicates the normalized minimization distance points from the point/edge/face that 
		 * p1 belongs to to antoher primmitve, where "l" denotes the length of the original distance before "d" is normalized. 
		 * And "l0" denotes the rest length of the selceted index. "isPointFace" indicates the type of altitude spring of whether it is 
		 * a point/face altitude spring or edge/edge altitude spring. "intp" is the interpolation weight for p1, p2, p3, p4. If the 
		 * spring is a edge/edge spring, the sum interpolation weight in the same primitive (point/face/edge) is always equal to 1 or -1.
		 * Another point of view of interpolation weight is where l * d = -(intp[0] * p1 + intp[1] * p2 + intp[2] * p3 + intp[3] * p4); The "signs"
		 * array indicates if a point is in the same primitive as p1, it will be 1.0 otherwise it will be -1.0;
		 * 
		 * @param sp The altitude spring inde
		 */
		struct AltitudeSpringInfo {
			int selectedIndex; ///< Which face/point or edge/edge springs are used. 

			Eigen::Vector3f d; ///< The normalized minimum distance vector (points from p1)

			float l, l0; ///< Current length and the rest length

			/// The interpolation weight of the closest point approach, so that
			/// l * d = -(intp[0] * p1 + intp[1] * p2 + intp[2] * p3 + intp[3] * p4)
			std::array<float, 4> intp;

			/// Indicating the primitive where the particle belongs to, if signs[i] = 1.0, then pi is 
			/// in the same primitive as p1; Otherwise signs[i] = -1.0
			std::array<float, 4> signs;

			Hair::Particle::Ptr p[4]; ///< The four particles
		};
		
		AltitudeSpringInfo getAltitudeSpringInfo(const AltitudeSpring * sp) {
			AltitudeSpringInfo ret;

			//FIXME: Fix the s, t computation in point/face springs
			//FIXME: Remove signs in the Altitude springs

			auto p1 = p(sp->i1), p2 = p(sp->i2), p3 = p(sp->i3), p4 = p(sp->i4);

			Eigen::Vector3f normals[7];
			Eigen::Vector3f vs[7];

			Eigen::Vector3f
				d12 = p2->pos - p1->pos,
				d13 = p3->pos - p1->pos,
				d14 = p4->pos - p1->pos,
				d23 = p3->pos - p2->pos,
				d24 = p4->pos - p2->pos,
				d34 = p4->pos - p3->pos;

			vs[0] = d13;
			vs[1] = d12;
			vs[2] = d12;

			vs[3] = d12;
			vs[4] = d12;
			vs[5] = d13;
			vs[6] = d14;

			normals[0] = d12.cross(d34);
			normals[1] = d13.cross(d24);
			normals[2] = d14.cross(d23);

			normals[3] = d23.cross(d24);
			normals[4] = d13.cross(d14);
			normals[5] = d12.cross(d14);
			normals[6] = d12.cross(d23);

			ret.selectedIndex = 0;
			float largestSquaredNormal = normals[0].squaredNorm();

			for (int i = 1; i < 7; ++i) {
				float squaredNormal = normals[i].squaredNorm();
				if (squaredNormal > largestSquaredNormal) {
					ret.selectedIndex = i;
					largestSquaredNormal = squaredNormal;
				}
			}

			// Set l, l0 and d (normalized) and make d in the right direction
			normals[ret.selectedIndex].normalize();
			ret.d = MathUtility::project(vs[ret.selectedIndex], normals[ret.selectedIndex]);
			ret.l = ret.d.norm();
			ret.d /= ret.l;
			if (ret.d.dot(vs[ret.selectedIndex]) < 0.0f)
				ret.d = -ret.d;
			ret.l0 = sp->l0s[ret.selectedIndex];

			// Compute the interpolation weights
			/*
			 * 0. Edge/Edge: {p1, p2} --> {p3, p4}
		     * 1. Edge/Edge: {p1, p3} --> {p2, p4}
			 * 2. Edge/Edge: {p1, p4} --> {p2, p3}
			 * 3. Point/Face: p1 --> {p2, p3, p4}
			 * 4. Point/Face: p2 --> {p1, p3, p4}
			 * 5. Point/Face: p3 --> {p1, p2, p4}
			 * 6. Point/Face: p4 --> {p1, p2, p3}
			 */			
		 	if (ret.selectedIndex < 3) {
		 		// Edge/edge spring
		 		std::pair<float, float> r;
		
		 		switch (ret.selectedIndex) {
		 		case 0:
		 			// 0. Edge/Edge: {p1, p2} --> {p3, p4}
		 			r = MathUtility::linetoLineDistanceClosestPointApproach(p1->pos, p2->pos, p3->pos, p4->pos);
		 			ret.intp = { 1 - r.first, r.first, r.second - 1, -r.second };
		 			ret.signs = { 1.0f, 1.0f, -1.0f, -1.0f };
		 			break;
		 		case 1:
		 			// 1. Edge/Edge: {p1, p3} --> {p2, p4}
		 			r = MathUtility::linetoLineDistanceClosestPointApproach(p1->pos, p3->pos, p2->pos, p4->pos);
		 			ret.intp = { 1 - r.first, r.second - 1, r.first, -r.second };
		 			ret.signs = { 1.0f, -1.0f, 1.0f, -1.0f };
		 			break;
		 		default:
		 			// 2. Edge/Edge: {p1, p4} --> {p2, p3}
		 			r = MathUtility::linetoLineDistanceClosestPointApproach(p1->pos, p4->pos, p2->pos, p3->pos);
		 			ret.intp = { 1 - r.first, r.second - 1, -r.second, r.first };
		 			ret.signs = { 1.0f, -1.0f, -1.0f, 1.0f };
		 			break;
		 		}
		 	}
		 	else {
		 		// Point/face spring
		 		Eigen::Vector3f p, o, d, e;
  
		 		switch (ret.selectedIndex) {
		 		case 3:
		 			// 3. Point/Face: p1 --> {p2, p3, p4}
		 			p = p1->pos;
		 			o = p2->pos;
		 			d = p3->pos;
		 			e = p4->pos;
		 			ret.signs = { 1.0f, -1.0f, -1.0f, -1.0f };
		 			break;
		 		case 4:
		 			// 4. Point/Face: p2 --> {p1, p3, p4}
		 			p = p2->pos;
		 			o = p1->pos;
		 			d = p3->pos;
		 			e = p4->pos;
		 			ret.signs = { 1.0f, -1.0f, 1.0f, 1.0f };
		 			break;
		 		case 5:
		 			// 5. Point/Face: p3 --> {p1, p2, p4}
		 			p = p3->pos;
		 			o = p1->pos;
		 			d = p2->pos;
		 			e = p4->pos;
		 			ret.signs = { 1.0f, 1.0f, -1.0f, 1.0f };
		 			break;
		 		default:
		 			// 6. Point/Face: p4 --> {p1, p2, p3}
		 			p = p4->pos;
		 			o = p1->pos;
		 			d = p2->pos;
		 			e = p3->pos;
		 			ret.signs = { 1.0f, 1.0f, 1.0f, -1.0f };
		 			break;
		 		}
  
		 		std::pair<float, float> st = MathUtility::pointToPlaneClosestPointApproach(p, o, d, e);
		 		float s = st.first;
		 		float t = st.second;
  
		 		switch (ret.selectedIndex) {
		 		case 3:
		 			ret.intp = { 1.0, s + t - 1.0f, -s, -t };
		 			break;
		 		case 4:
		 			ret.intp = { 1 - s - t, -1.0f, s, t };
		 			break;
		 		case 5:
		 			ret.intp = { 1 - s - t, s, -1.0f, t };
		 			break;
		 		default:
		 			ret.intp = { 1 - s - t, s, t, -1.0f };
		 			break;
		 		}
		 	}

			// Compute the spring force
			ret.p[0] = p1;
			ret.p[1] = p2;
			ret.p[2] = p3;
			ret.p[3] = p4;
  
		 	return ret;
		 }
	};
}