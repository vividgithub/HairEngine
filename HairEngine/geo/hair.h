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


#include "../util/fileutil.h"
#include "../precompiled/precompiled.h"
#include "../util/eigenutil.h"

namespace HairEngine {

	class Hair;
	class Integrator;
	class SelleMassSpringSolverBase;

	std::ostream & operator<<(std::ostream & os, const Hair & hair);

	/*
	 * A representation for the hair geometry for the solver
	 */
	class Hair {

		friend class Integrator;
		friend class SelleMassSpringSolverBase;

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

			size_t localIndex; ///< Local index for the particle in the strand
			size_t globalIndex; ///< Global index for the particle in the whole hair geometry

			Strand *strandPtr; ///< The associated strand pointer

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
			 * Check whether it is the first particle in the strand
			 *
			 * @return True if it is the first particle and otherwise false
			 */
			bool isStrandRoot() const;

			/**
			 * Check whether it is the last particle in the strand
			 *
			 * @return True if it is the last particle in the strand and otherwise false
			 */
			bool isStrandTip() const;

			/**
			 * Initialization of the particle
			 *
			 * @param restPos The rest position of the particle position
			 * @param pos Current position of the particle
			 * @param vel Current velocity of the particle
			 * @param impulse Current impluse of the particle
			 * @param localIndex The local index in the strand
			 * @param globalIndex The global index in the strand
			 * @param strandPtr The strand pointer of which the particle is in
			 */
			Particle(const Eigen::Vector3f &restPos, const Eigen::Vector3f &pos, const Eigen::Vector3f &vel,
			         const Eigen::Vector3f &impulse, size_t localIndex, size_t globalIndex, Strand *strandPtr)
					: restPos(restPos), pos(pos), vel(vel), impulse(impulse), localIndex(localIndex),
					  globalIndex(globalIndex), strandPtr(strandPtr) {}
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

			size_t localIndex; ///< Local index for the segment in the strand
			size_t globalIndex; ///< Global index for the segment in the whole hair geometry

			/**
			 * Get the strand pointer which the segment belongs to
			 *
			 * @return The strand pointer
			 */
			Strand *strandPtr() {
				return p1->strandPtr;
			}

			/**
			 * Get the strand pointer(const version) which the segment belongs to
			 *
			 * @return The strand pointer
			 */
			const Strand *strandPtr() const {
				return p1->strandPtr;
			}

			/**
			 * Current direction d = p2->pos - p1->pos
			 * 
			 * @return The direction d
			 */
			Eigen::Vector3f d() const {
				return p2->pos - p1->pos;
			}

			/**
			 * Constructor
			 *
			 * @param p1 The first particle pointer
			 * @param p2 The second particle pointer
			 * @param localIndex The local index of the segment in the strand
			 * @param globalIndex The global index of the segment in the strand
			 */
			Segment(Particle *p1, Particle *p2, size_t localIndex, size_t globalIndex) :
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
				size_t nparticle;
			} particleInfo;

			struct
			{
				Segment *beginPtr;
				Segment *endPtr;
				size_t nsegment;
			} segmentInfo;

			size_t index; ///< The index in the global hair geometry
			Hair *hairPtr; ///< Point to the hair geometry that it belongs to

			/**
			 * Constructor with a single index in the hair geometry, we initialize the strand with empty vectors
			 * for the particle pointers and segments pointers.
			 *
			 * @param index The index of strand
			 */
			Strand(size_t index, Hair* hairPtr, Particle *particleBeginPtr, Particle *particleEndPtr, Segment *segmentBeginPtr, Segment *segmentEndPtr): 
				index(index), hairPtr(hairPtr),
				particleInfo{ particleBeginPtr, particleEndPtr, 0 }, 
				segmentInfo{ segmentBeginPtr, segmentEndPtr, 0 } {

				particleInfo.nparticle = static_cast<size_t>(particleEndPtr - particleBeginPtr);
				segmentInfo.nsegment = static_cast<size_t>(segmentEndPtr - segmentBeginPtr);
			}
		};

	HairEngine_Protected:

		Particle *particles = nullptr; ///< All particles in the hair geometry
		size_t nparticle = 0; ///< Number of particles

		Segment *segments = nullptr; ///< All segments in the hair geometry
		size_t nsegment = 0; ///< Number of segments

		Strand *strands = nullptr; ///< All strands in the hair geometry
		size_t nstrand = 0; ///< Number of strands

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
		 * Deconstructor
		 */
		virtual ~Hair()
		{
			HairEngine_SafeDeleteArray(particles);
			HairEngine_SafeDeleteArray(segments);
			HairEngine_SafeDeleteArray(strands);
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
		 * Get the end pointer of the particles
		 */
		Particle *particleEnd() { return particles + nparticle; }

		/**
		* Get the end pointer of the particles
		*/
		const Particle *particleEnd() const { return particles + nparticle; }

		/**
		 * Get the end pointer of the segments
		 */
		Segment *segmentEnd() { return segments + nsegment; }

		/**
		 * Get the end pointer of the segments
		 */
		const Segment *segmentEnd() const { return segments + nsegment; }

		/**
		 * Get the end pointer of the strands
		 */
		Strand *strandEnd() { return strands + nstrand; }

		/**
		 * Get the end pointer of the strands
		 */
		const Strand *strandEnd() const { return strands + nstrand;  }

	HairEngine_Protected:

		/**
		 * Helper function for constructor, takes two iterator. One is the position iterator, which is used to
		 * iterate over all the particle positions. Another is the strand size iterator, which indicates how many
		 * particles is in the strand.
		 *
		 * @tparam RestPositionIterator (*RestPositionIterator) should return a Eigen::Vector3f type.
		 * @tparam StrandSizeIterator (*StrandSizeIterator) should return a size_t or other type that could cast to it.
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

				const size_t nparticleInStrand = *strandSizeIt;

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

				for (size_t i = 0; i < nparticleInStrand; ++i) {
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
						nparticle, strandPtr
					);

					// Check whether we could create a segment
					if (i > 0) {
						// Initialize the segment
						auto segmentPtr = segments + nsegment;
						segmentAllocator.construct(
							segmentPtr,
							// Currently, nparticle is not upadted so (particles + nparticle) points to the last allocated particle 
							particlePtr + nparticle - 1, particlePtr + nparticle, 
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

			init(particlePositions.begin(), strandSizes.begin(), strandSizes.end());
		}

		/**
		 * Write the hair geometry to .hair file format. We only the current position to the .hair file format.
		 *
		 * @param filePath The file path for the .hair file format
		 */
		void writeToFile(const std::string & filePath) const {
			std::ofstream fout(filePath, std::ios::out | std::ios::binary);

			FileUtility::binaryWriteInt32(fout, static_cast<int32_t>(nparticle));
			for (size_t i = 0; i < nparticle; ++i)
				FileUtility::binaryWriteVector3f(fout, particles[i].pos);

			FileUtility::binaryWriteInt32(fout, static_cast<int32_t>(nstrand));
			for (size_t i = 0; i < nstrand; ++i)
				FileUtility::binaryWriteInt32(fout, static_cast<int32_t>(strands[i].particleInfo.nparticle));
		}
	};
}
