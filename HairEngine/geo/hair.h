//
// Created by vivi on 16/06/2018.

#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <fstream>
#include <ostream>
#include <iostream>


#include "../util/fileutil.h"
#include "../precompiled/precompiled.h"
#include "../util/eigenutil.h"

namespace HairEngine {

	class Hair;

	std::ostream & operator<<(std::ostream & os, const Hair & hair);

	/*
	 * A representation for the hair geometry for the solver
	 */
	class Hair {
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

			std::vector<Particle::Ptr> particlePtrs; ///< The particle pointers in the strand
			std::vector<Segment::Ptr> segmentPtrs; ///< The segment pointers in the strand

			size_t index; ///< The index in the global hair geometry

			/**
			 * Constructor with a single index in the hair geometry, we initialize the strand with empty vectors
			 * for the particle pointers and segments pointers.
			 *
			 * @param index The index of strand
			 */
			Strand(size_t index): index(index) {}
		};

	HairEngine_Protected:

		std::vector<Particle> particles; ///< All particles in the hair geometry
		std::vector<Segment> segments; ///< All segments in the hair geometry
		std::vector<Strand> strands; ///< All strands in the hair geometry

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
		 * Construction from .hair file format.
		 *
		 * @param filePath The file path from the .hair file.
		 * @param affine The initial affine transform.
		 */
		Hair(const std::string & filePath, const Eigen::Affine3f & affine = Eigen::Affine3f::Identity()) {
			auto hairFile = std::fstream(filePath, std::ios::in | std::ios::binary);
			init(hairFile, affine);
		}

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

			// Clear current contents
			particles.clear();
			segments.clear();
			strands.clear();

			// Count the size of strand and particle in order to alloc the space, otherwise it
			size_t nParticle = 0, nStrand = 0;
			for (auto strandSizeIt = strandSizeBegin; strandSizeIt != strandSizeEnd; ++strandSizeIt) {
				nParticle += *strandSizeIt;
				++nStrand;
			}

			// Reverse the size of the particles, segments, strands array, so that we could get the pointer since
			// the vector will not be reallocted
			particles.reserve(nParticle);
			segments.reserve(nParticle - nStrand);
			strands.reserve(nStrand);

			// Initialize particles, segments, strands
			size_t strandGloablIndex = 0, particleGlobalIndex = 0, segmentGlobalIndex = 0;
			auto posIt = posBegin;

			// Iterate over all the strands
			for (auto strandSizeIt = strandSizeBegin; strandSizeIt != strandSizeEnd; ++strandSizeIt) {

				// Create a strand
				strands.emplace_back(strandGloablIndex);
				auto & strand = strands.back();

				for (size_t i = 0; i < *strandSizeIt; ++i) {
					// Initialize the particle
					Eigen::Vector3f pos = affine * (*posIt);
					particles.emplace_back( pos, pos, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), i, particleGlobalIndex, &strand);

					// Add to the strand
					strand.particlePtrs.push_back(&particles.back());

					++particleGlobalIndex;

					// Check whether we could create a segment
					if (i > 0) {
						// Initialize the segment
						segments.emplace_back( &particles[particles.size() - 2], &particles[particles.size() - 1], i - 1, segmentGlobalIndex);

						// Add to the strand
						strand.segmentPtrs.push_back(&segments.back());
						++segmentGlobalIndex;
					}

					++posIt;
				}

				++strandGloablIndex;
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
			for (size_t i = 0; i < particleSize; ++i)
				particlePositions.push_back(FileUtility::binaryReadVector3f(is));

			// Read strand sizes
			strandSize = FileUtility::binaryReadInt32(is);
			for (size_t i = 0; i < strandSize; ++i)
				strandSizes.push_back(FileUtility::binaryReadInt32(is));

			init(particlePositions.begin(), strandSizes.begin(), strandSizes.end());
		}

		/**
		 * Write the hair geometry to .hair file format. We only the current position to the .hair file format.
		 *
		 * @param filePath The file path for the .hair file format
		 */
		void writeToFile(const std::string & filePath) {
			std::ofstream fout(filePath, std::ios::out | std::ios::binary);

			FileUtility::binaryWriteInt32(fout, static_cast<int32_t>(particles.size()));
			for (const auto & p : particles)
				FileUtility::binaryWriteVector3f(fout, p.pos);
			FileUtility::binaryWriteInt32(fout, static_cast<int32_t>(strands.size()));
			for (const auto & st : strands)
				FileUtility::binaryWriteInt32(fout, static_cast<int32_t>(st.particlePtrs.size()));
		}
	};

	std::ostream & operator<<(std::ostream & os, const Hair::Segment & segment);
	std::ostream & operator<<(std::ostream & os, const Hair::Particle & particle);
	std::ostream & operator<<(std::ostream & os, const Hair::Strand & strand);
}

