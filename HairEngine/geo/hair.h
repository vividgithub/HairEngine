//
// Created by vivi on 16/06/2018.
//

#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <fstream>
#include <ostream>
#include <iostream>
#include <memory>
#include <algorithm>


#include "../util/fileutil.h"
#include "../precompiled/precompiled.h"
#include "../util/eigenutil.h"
#include "HairEngine/HairEngine/util/mathutil.h"

namespace HairEngine {

	class Hair;
	class Integrator;
	class SelleMassSpringSolverBase;
	class HairVisualizer;
	class PositionCommiter;
	class Solver;
	class SelleMassSpringImplicitSolver;
	class SegmentKNNSolver;
	class SegmentKNNSolverVisualizer;
	class HairContactsImpulseSolver;

	std::ostream & operator<<(std::ostream & os, const Hair & hair);

	/*
	 * A representation for the hair geometry for the solver
	 */
	class Hair {

		friend class Integrator;
		friend class SelleMassSpringSolverBase;
		friend class HairVisualizer;
		friend class PositionCommiter;
		friend class Solver;
		friend class SelleMassSpringImplicitSolver;
		friend class SegmentKNNSolver;
		friend class SegmentKNNSolverVisualizer;
		friend class HairContactsImpulseSolver;
		friend class HairContactsAndCollisionImpulseSolverVisualizer;
		friend class HairContactsImpulseSolverOld;
		friend class CollisionImpulseSolver;
		friend class HairContactsImpulseSolver;
		friend class SDFCollisionSolver;
		friend class CudaMemoryConverter;
		friend class CudaMemoryInverseConverter;

	HairEngine_Public:
		struct Strand;

		/**
		 * Representation of single hair particle in the hair geometry
		 */
		struct Particle {

			typedef Particle *Ptr;

			Eigen::Vector3f restPos; ///< The rest position
			Eigen::Vector3f pos; ///< The position of current frame
			Eigen::Vector3f vel; ///< The velocity of current frame
			Eigen::Vector3f impulse; ///< The impulse of current frame

			int localIndex; ///< Local index for the particle in the strand
			int globalIndex; ///< Global index for the particle in the whole hair geometry

			int strandIndex; ///< The associated strand's index in the hair geometry

			/**
			 * Predict the particles' position after time t
			 *
			 * @param t The specific time
			 * @return The predicted position
			 */
			Eigen::Vector3f predictedPos(float t) const {
				return pos + t * vel;
			}

			/**
			 * Initialization of the particle
			 *
			 * @param restPos The rest position of the particle position
			 * @param pos Current position of the particle
			 * @param vel Current velocity of the particle
			 * @param impulse Current impluse of the particle
			 * @param localIndex The local index in the strand
			 * @param globalIndex The global index in the strand
			 * @param strandIndex The strand index of which the particle is in
			 */
			Particle(const Eigen::Vector3f &restPos, const Eigen::Vector3f &pos, const Eigen::Vector3f &vel,
			         const Eigen::Vector3f &impulse, int localIndex, int globalIndex, int strandIndex)
					: restPos(restPos), pos(pos), vel(vel), impulse(impulse), localIndex(localIndex),
					  globalIndex(globalIndex), strandIndex(strandIndex) {}
		};


		/**
		 * Representation of single segment in the hair geometry. In the hair geometry,
		 * a "Segment" is the line segment which connects two adjacent particles in the strand.
		 * A strand consits of several segments. In the "Segment", we store two adjacent particle
		 * pointer in the same strand.
		 */
		struct Segment {
			typedef Segment *Ptr;

			Particle::Ptr p1, p2; ///< Two adjacent particle pointers

			int localIndex; ///< Local index for the segment in the strand
			int globalIndex; ///< Global index for the segment in the whole hair geometry

			/**
			 * Current direction d = p2->pos - p1->pos
			 * 
			 * @return The direction d
			 */
			Eigen::Vector3f d() const {
				return p2->pos - p1->pos;
			}

			/**
			 * Get the midpoint of the segment
			 * 
			 * @return The midpoint coordinate
			 */
			Eigen::Vector3f midpoint() const {
				return 0.5f * (p2->pos + p1->pos);
			}

			/**
			 * Return the strand index of the segment
			 * 
			 * @return The strand index of the segment
			 */
			int strandIndex() const {
				return p1->strandIndex;
			}

			/**
			 * Get the lerp position p1->pos + (p2->pos - p1->pos) * t
			 * 
			 * @param t The interpolation weight 
			 * @return The interpolation point
			 */
			Eigen::Vector3f lerpPos(float t) const {
				return MathUtility::lerp(p1->pos, p2->pos, t);
			}

			/**
			 * Get the lerp velocity p1->vel + (p2->vel - p1->vel) * t
			 * 
			 * @param t The interpolation weight
			 * @return The interpolation velocity
			 */
			Eigen::Vector3f lerpVel(float t) const {
				return MathUtility::lerp(p1->vel, p2->vel, t);
			}

			/**
			 * Constructor
			 *
			 * @param p1 The first particle pointer
			 * @param p2 The second particle pointer
			 * @param localIndex The local index of the segment in the strand
			 * @param globalIndex The global index of the segment in the strand
			 */
			Segment(Particle *p1, Particle *p2, int localIndex, int globalIndex) :
					p1(p1), p2(p2),
					localIndex(localIndex),
					globalIndex(globalIndex) {}
		};


		/**
		 * Representation of the strand in the hair geometry. A hair geometry consists of several strands.
		 * A strand consists of several particles and segments.
		 */
		struct Strand {
			typedef Strand *Ptr;

			struct {
				Particle *beginPtr;
				Particle *endPtr;
				int nparticle;
			} particleInfo;

			struct
			{
				Segment *beginPtr;
				Segment *endPtr;
				int nsegment;
			} segmentInfo;

			int index; ///< The index in the global hair geometry
			Hair *hairPtr; ///< Point to the hair geometry that it belongs to

			/**
			 * Constructor with a single index in the hair geometry, we initialize the strand with empty vectors
			 * for the particle pointers and segments pointers.
			 *
			 * @param index The index of strand
			 */
			Strand(int index, Hair* hairPtr, Particle *particleBeginPtr, Particle *particleEndPtr, Segment *segmentBeginPtr, Segment *segmentEndPtr): 
				index(index), hairPtr(hairPtr),
				particleInfo{ particleBeginPtr, particleEndPtr, 0 }, 
				segmentInfo{ segmentBeginPtr, segmentEndPtr, 0 } {

				particleInfo.nparticle = static_cast<int>(particleEndPtr - particleBeginPtr);
				segmentInfo.nsegment = static_cast<int>(segmentEndPtr - segmentBeginPtr);
			}
		};

		Particle *particles = nullptr; ///< All particles in the hair geometry
		int nparticle = 0; ///< Number of particles

		Segment *segments = nullptr; ///< All segments in the hair geometry
		int nsegment = 0; ///< Number of segments

		Strand *strands = nullptr; ///< All strands in the hair geometry
		int nstrand = 0; ///< Number of strands

	HairEngine_Public:

		/**
		 * Constructor from a input stream and a affine transform
		 *
		 * @param is The input stream(binary). The input stream should be composed first by particleCount(int32_t),
		 * particlePositions(particleSize * float * 3), strandCount(int32_t), strandSizes (int32_t * strandCount).
		 *
		 * @param affine The affine transform. We could provide an transform to all the positions from the input
		 * stream, so that the rest positions is caculated as "affine * readingPosition".
		 */
		Hair(std::istream & is, const Eigen::Affine3f & affine = Eigen::Affine3f::Identity()) {
			init(is, affine);
		}

		/**
		 * Copy constructor
		 */
		Hair(const Hair & rhs) {
			nparticle = rhs.nparticle;
			nstrand = rhs.nstrand;
			nsegment = rhs.nsegment;

			HairEngine_AllocatorAllocate(particles, nparticle);
			HairEngine_AllocatorAllocate(strands, nstrand);

			std::copy(rhs.particles, rhs.particleEnd(), particles);
			std::copy(rhs.strands, rhs.strandEnd(), strands);
		}

		/**
		 * Move constructor
		 */
		Hair(Hair && rhs) {
			std::swap(nparticle, rhs.nparticle);
			std::swap(nsegment, rhs.nsegment);
			std::swap(nstrand, rhs.nstrand);
			std::swap(particles, rhs.particles);
			std::swap(segments, rhs.segments);
			std::swap(strands, rhs.strands);
		}

		/**
		 * Deconstructor
		 */
		virtual ~Hair()
		{
			HairEngine_AllocatorDeallocate(particles, nparticle);
			HairEngine_AllocatorDeallocate(segments, nsegment);
			HairEngine_AllocatorDeallocate(strands, nstrand);
		}

		/**
		 * Construction from .hair file format.
		 *
		 * @param filePath The file path from the .hair file.
		 * @param affine The initial affine transform.
		 */
		Hair(const std::string & filePath, const Eigen::Affine3f & affine = Eigen::Affine3f::Identity()) {
			auto hairFile = std::fstream(filePath, std::ios::in | std::ios::binary);
			init(hairFile, affine);
		}

		/**
		 * A wrapper for init function
		 */
		template <class RestPositionIterator, class StrandSizeIterator>
		Hair(const RestPositionIterator & posBegin,
		     const StrandSizeIterator & strandSizeBegin,
		     const StrandSizeIterator & strandSizeEnd,
		     const Eigen::Affine3f & affine = Eigen::Affine3f::Identity()) {
			init<RestPositionIterator, StrandSizeIterator>(posBegin, strandSizeBegin, strandSizeEnd);
		}

		/**
		 * Get the end pointer of the particles
		 */
		Particle *particleEnd() const { return particles + nparticle; }

		/**
		* Get the end pointer of the particles
		*/
		//const Particle *particleEnd() const { return particles + nparticle; }

		/**
		 * Get the end pointer of the segments
		 */
		Segment *segmentEnd() const { return segments + nsegment; }

		/**
		 * Get the end pointer of the segments
		 */
		//const Segment *segmentEnd() const { return segments + nsegment; }

		/**
		 * Get the end pointer of the strands
		 */
		Strand *strandEnd() const { return strands + nstrand; }

		/**
		 * Get the end pointer of the strands
		 */
		//const Strand *strandEnd() const { return strands + nstrand;  }

		/**
		 * Resampling some strands specified by the IndexIterator. IndexIterator should 
		 * yield the index for the strand that want to fetch from the orignial hair geometry.
		 * 
		 * @param begin The begin index iterator
		 * @param end The end index iterator
		 * @return A resampled hair geometry
		 */
		template <class IndexIterator>
		Hair resample(const IndexIterator & begin, const IndexIterator & end) const {
			std::vector<Eigen::Vector3f> positions;
			std::vector<int> strandSizes;

			for (auto it = begin; it != end; ++it) {
				const Strand & strand = strands[*it];

				strandSizes.push_back(strand.particleInfo.nparticle);
				for (auto p = strand.particleInfo.beginPtr; p != strand.particleInfo.endPtr; ++p) {
					positions.push_back(p->pos);
				}
			}

			return Hair(positions.begin(), strandSizes.begin(), strandSizes.end());
		}

		/**
		 * Resampling the hair. The resampled hair should have appoximate "nstrand / sampleRate" strands.
		 * 
		 * @param sampleRate The rate for sampling
		 * @return Resampled hair
		 */
		Hair resample(const int sampleRate) {
			std::vector<int> sampledStrandIndices;
			for (int i = 0; i < nstrand; i += sampleRate) {
				sampledStrandIndices.push_back(i);
			}

			return resample(sampledStrandIndices.begin(), sampledStrandIndices.end());
		}

		/**
		* Write hair information into specific stream
		*
		* @param os The ostream to write to
		*/
		void stream(std::ostream & os) const {
			FileUtility::binaryWriteInt32(os, static_cast<int32_t>(nparticle));
			for (int i = 0; i < nparticle; ++i)
				FileUtility::binaryWriteVector3f(os, particles[i].pos);

			FileUtility::binaryWriteInt32(os, static_cast<int32_t>(nstrand));
			for (int i = 0; i < nstrand; ++i)
				FileUtility::binaryWriteInt32(os, static_cast<int32_t>(strands[i].particleInfo.nparticle));
		}

		/**
		* Write the hair geometry to .hair file format. We only the current position to the .hair file format.
		*
		* @param filePath The file path for the .hair file format
		*/
		void writeToFile(const std::string & filePath) const {
			std::ofstream fout(filePath, std::ios::out | std::ios::binary);
			stream(fout);
		}

	HairEngine_Protected:

		/**
		 * Helper function for constructor, takes two iterator. One is the position iterator, which is used to
		 * iterate over all the particle positions. Another is the strand size iterator, which indicates how many
		 * particles is in the strand.
		 *
		 * @tparam RestPositionIterator (*RestPositionIterator) should return a Eigen::Vector3f type.
		 * @tparam StrandSizeIterator (*StrandSizeIterator) should return a int or other type that could cast to it.
		 * @param posBegin The begin of the position iterator. We do not use the posEnd since we could get the total
		 * particle count from the strand size iterator.
		 * @param strandSizeBegin The begin of the strand size iterator.
		 * @param strandSizeEnd The end of the strand size iterator.
		 * @param affine The initial affine transform.
		 */
		template <class RestPositionIterator, class StrandSizeIterator>
		void init(const RestPositionIterator & posBegin,
		          const StrandSizeIterator & strandSizeBegin,
		          const StrandSizeIterator & strandSizeEnd,
		          const Eigen::Affine3f & affine = Eigen::Affine3f::Identity()) {

			auto particleAllocator = std::allocator<Particle>();
			auto strandAllocator = std::allocator<Strand>();
			auto segmentAllocator = std::allocator<Segment>();

			// Count the size of strand and particle in order to alloc the space
			nparticle = nstrand = 0;
			for (auto strandSizeIt = strandSizeBegin; strandSizeIt != strandSizeEnd; ++strandSizeIt) {
				nparticle += *strandSizeIt;
				++nstrand;
			}
			nsegment = nparticle - nstrand;

			// Allocate the space for particles, segments and strands
			particles = particleAllocator.allocate(nparticle);
			strands = strandAllocator.allocate(nstrand);
			segments = segmentAllocator.allocate(nsegment);


			// Reset the nparticle, nstrand and nsegment, we will increase in the construction
			nsegment = nparticle = nstrand = 0;

			auto posIt = posBegin;

			// Iterate over all the strands
			for (auto strandSizeIt = strandSizeBegin; strandSizeIt != strandSizeEnd; ++strandSizeIt) {

				const int nparticleInStrand = *strandSizeIt;

				// Create a strand
				auto strandPtr = strands + nstrand;
				strandAllocator.construct(
					strandPtr,
					nstrand, this,
					particles + nparticle,
					particles + nparticle + nparticleInStrand,
					segments + nsegment,
					segments + nsegment + nparticleInStrand - 1
				);

				for (int i = 0; i < nparticleInStrand; ++i) {
					// Initialize the particle
					Eigen::Vector3f pos = affine * (*posIt);

					// Create a particle
					auto particlePtr = particles + nparticle;
					particleAllocator.construct(
						particlePtr, 
						pos,
						pos,
						Eigen::Vector3f::Zero(),  
						Eigen::Vector3f::Zero(),
						i, 
						nparticle, strandPtr->index
					);

					// Check whether we could create a segment
					if (i > 0) {
						// Initialize the segment
						auto segmentPtr = segments + nsegment;
						segmentAllocator.construct(
							segmentPtr,
							// Currently, nparticle is not upadted so (particles + nparticle) points to the last allocated particle 
							particlePtr - 1, particlePtr, 
							i - 1, nsegment
						);

						++nsegment;
					}

					++posIt;
					++nparticle;
				}

				++nstrand;
			}
		};

		/**
		 * Helper function for the constructor.
		 * @param is Same as the constructor.
		 * @param affine Same as the constructor.
		 */
		void init(std::istream & is, const Eigen::Affine3f & affine = Eigen::Affine3f::Identity()) {
			//Ready to read from the stream
			std::vector<Eigen::Vector3f> particlePositions;
			std::vector<int32_t> strandSizes;

			int32_t particleSize;
			int32_t strandSize;

			// Read particle positions
			particleSize = FileUtility::binaryReadInt32(is);
			for (int32_t i = 0; i < particleSize; ++i)
				particlePositions.push_back(FileUtility::binaryReadVector3f(is));

			// Read strand sizes
			strandSize = FileUtility::binaryReadInt32(is);
			for (int32_t i = 0; i < strandSize; ++i)
				strandSizes.push_back(FileUtility::binaryReadInt32(is));

			init(particlePositions.begin(), strandSizes.begin(), strandSizes.end(), affine);
		}


	};
}
