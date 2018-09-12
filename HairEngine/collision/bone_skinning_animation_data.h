//
// Created by vivi on 2018/9/12.
//

#pragma once

#include <vector>
#include <array>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <Eigen/Eigen>
#include <Eigen/StdVector>


#include "../precompiled/precompiled.h"
#include "../util/fileutil.h"
#include "../util/mathutil.h"
#include "../util/eigenutil.h"

namespace HairEngine {
	class BoneSkinningAnimationData {

		using Affine3fUnaligned = Eigen::Transform<float, 3, Eigen::Affine, Eigen::DontAlign>;

		friend class BoneSkinningAnimationDataVisualizer;

	HairEngine_Public:

		/**
		 * Read the bone skinning data
		 * @param filePath The file path for opening the bone skinning data
		 */
		BoneSkinningAnimationData(const std::string & filePath) {
			std::ifstream fin(filePath, std::ios::in | std::ios::binary);

			if (!fin) {
				throw HairEngineIOException();
			}

			init(fin);
		};

		/**
		 * Update the point position to specified time. Result will be stored in
		 * the poses array
		 * @param time The time to update
		 */
		void update(float time) {

			std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f>> ctransforms(nbone); // Current transforms

			// Get the transform for all bone skinning
			float timeInFrame = time / frameTimeInterval;
			float alpha = timeInFrame - std::floorf(timeInFrame);

			if (timeInFrame <= 0.0f || timeInFrame >= nframe - 1) {
				// If it is out of range, then copy the first or last transform into ctransforms
				int offset = (timeInFrame <= 0.0f) ? 0 : (nframe - 1) * nbone;
				for (int i = 0; i < nbone; ++i) {
					ctransforms[i] = transforms[offset + i];
				}
			}
			else {
				// Lerp the transform into ctransform
				int prevOffset = static_cast<int>(timeInFrame) * nbone;
				int nextOffset = prevOffset + nbone;

				for (int i = 0; i < nbone; ++i) {
					Eigen::Affine3f prevTransform = transforms[prevOffset + i];
					Eigen::Affine3f nextTransform = transforms[nextOffset + i];
					ctransforms[i] = MathUtility::lerp(alpha, prevTransform, nextTransform);
				}
			}

			// Lerp the position
			for (int i = 0; i < npoint; ++i) {

				poses[i] = Eigen::Vector3f::Zero();
				int offset = i * nbonePerPoint;

				for (int bi = 0; bi < nbonePerPoint; ++bi) {
					const auto & bw = boneWeights[offset + bi];

					if (bw.boneIndex < 0)
						break;

					poses[i] += ctransforms[bw.boneIndex] * restPoses[i] * bw.weight;
				}
			}
		}

		/**
		 * Print the summary of the BoneSkinningAnimationData
		 *
		 * @param os The ostream for output
		 */
		void printSummary(std::ostream & os) {
			os << "BoneSkinningAnmationData:" << '\n';
			os << "Bone number: " << nbone << '\n';
			os << "Point number: " << npoint << '\n';
			os << "Primitive number: " << nprim << '\n';
			os << "Bone data per point: " << nbonePerPoint << '\n';
			os << "Frame: " << nframe << '\n';
			os << "Frame time interval: " << frameTimeInterval << '\n';

			const auto & getBoundingBoxAtTime = [this] (float time) -> Eigen::AlignedBox3f {
				update(time);
				Eigen::AlignedBox3f bbox(poses[0]);
				for (const auto & pos : poses) {
					bbox.extend(pos);
				}
				return bbox;
			};

			if (npoint > 0) {

				float minWeight = boneWeights[0].weight, maxWeight = boneWeights[0].weight;
				for (int i = 1; i < npoint * nbonePerPoint; ++i) {
					minWeight = std::min(minWeight, boneWeights[i].weight);
					maxWeight = std::max(maxWeight, boneWeights[i].weight);
				}

				os << "Min weight: " << minWeight << '\n';
				os << "Max weight: " << maxWeight << '\n';

				Eigen::AlignedBox3f bbox;

				bbox = getBoundingBoxAtTime(0.0f);
				os << "Skinning bounding box for first transform: ["
				<< EigenUtility::toString(bbox.min()) << ',' << EigenUtility::toString(bbox.max()) << "]\n";

				bbox = getBoundingBoxAtTime((nframe - 1) * frameTimeInterval);
				os << "Skinning bounding box for last transform: ["
				   << EigenUtility::toString(bbox.min()) << ',' << EigenUtility::toString(bbox.max()) << "]\n";
			}

			os << std::flush;
		}

		/**
		 * Get number of frame
		 * @return The numebr of frame
		 */
		int getFrameCount() const {
			return nframe;
		}

		/**
		 * Get the frame time interval
		 * @return The time interval
		 */
		float getFrameTimeInterval() const {
			return frameTimeInterval;
		}

		/**
		 * Get the total time interval
		 * @return The total time interval
		 */
		float getTotalTime() const {
			return nframe * frameTimeInterval;
		}

		/**
		 * Get the point position, call after "update"
		 * @param pointIdx The index of a point
		 * @return The point position for the specified index
		 */
		Eigen::Vector3f getPointPos(int pointIdx) const {
			return poses[pointIdx];
		}

		/**
		 * Get the triangle indices for an givien primitive index
		 *
		 * @param primIdx The primitive (triangle) index
		 * @return
		 */
		 std::array<int, 3> getPrimPointIndices(int primIdx) const {
		 	return primIndices[primIdx];
		 }

	HairEngine_Protected:
		int nbone; ///< Number of bones
		int npoint; ///< Number of mesh points
		int nprim; ///< Number of mesh primitives (triangle)

		int nbonePerPoint; ///< Number of bone skinning data per point

		int nframe; ///< Number of animation frame
		float frameTimeInterval; ///< The time interval of a frame

		std::vector<std::array<int, 3>> primIndices; ///< The point indices for each triangle
		std::vector<Eigen::Vector3f> restPoses; ///< The rest position of the points

		std::vector<Eigen::Vector3f> poses; ///< The current positions

		/// The bone skinning interpolation weighs
		/// We store the bone weights in a continuous memory, which means size equals to [npoint * nbone]
		struct BoneWeight {
			int boneIndex;
			float weight;

			BoneWeight(int boneIndex, float weight): boneIndex(boneIndex), weight(weight) {}
		};
		std::vector<BoneWeight> boneWeights;

		/// The transform for each frame
		/// Store in continuous memory, which means the size qeuals to [nframe * nbone]
		std::vector<Affine3fUnaligned> transforms;

		/**
		 * Initialize the bone skinning from a binary stream (normally a fstream)
		 * @param is The input binary stream
		 */
		void init(std::istream & is) {

			const auto readInt = [&is] () -> int {
				HairEngine_ThrowExceptionIf(!is, HairEngineIOException());
				return static_cast<int>(FileUtility::binaryReadInt32(is));
			};

			const auto readReal = [&is] () -> float {
				HairEngine_ThrowExceptionIf(!is, HairEngineIOException());
				return FileUtility::binaryReadReal32(is);
			};

			nbone = readInt();
			npoint = readInt();
			nprim = readInt();
			nbonePerPoint = readInt();

			nframe = readInt();
			frameTimeInterval = readReal();

			primIndices.reserve(static_cast<size_t>(nprim));
			for (int i = 0; i < nprim; ++i) {
				primIndices.emplace_back();

				auto & indices = primIndices.back();
				indices = { readInt(), readInt(), readInt() };

				// Check inside the primitive
				HairEngine_ThrowExceptionIf(
						indices[0] < 0 || indices[0] >= npoint || indices[1] < 0
						|| indices[1] >= npoint || indices[2] < 0 || indices[2] >= npoint,
						HairEngineIOException()
				);
			}

			restPoses.reserve(static_cast<size_t>(npoint));
			for (int i = 0; i < npoint; ++i) {
				restPoses.emplace_back(readReal(), readReal(), readReal());
			}

			poses.resize(static_cast<size_t>(npoint));

			boneWeights.reserve(static_cast<size_t>(npoint * nbonePerPoint));
			for (int i = 0; i < npoint * nbonePerPoint; ++i) {
				boneWeights.emplace_back(readInt(), readReal());
			}

			const auto & readSingleFrameTransform= [this, &readReal] () {

				for (int i = 0; i < nbone; ++i) {
					transforms.emplace_back();
					auto & mat = transforms.back().matrix();
					for (int r = 0; r < 4; ++r)
						for (int c = 0; c < 4; ++c)
							mat(r, c) = readReal();

					HairEngine_DebugIf {
						std::cout << "Bone " << i << ":" << std::endl;
						std::cout << mat << std::endl;
					};
				}
			};

			// FIXME: Inverse after interpolation
			// We store rest transform into the transforms array so that we could reuse the "update" function
			transforms.reserve(nbone);
			readSingleFrameTransform();
			for (auto & transform : transforms) {
				transform = transform.inverse(Eigen::Affine);
			}
			update(-1.0f); // Use negative value to force update to the first frame
			std::copy(poses.begin(), poses.end(), restPoses.begin());

			// Get the transforms
			transforms.clear();
			transforms.reserve(nbone * nframe);
			for (int i = 0; i < nframe; ++i) {
				std::cout << "==============" << i << "==============" << std::endl;
				readSingleFrameTransform();
				std::cout << "============================" << std::endl;
			}

			HairEngine_DebugIf {
				printSummary(std::cout);
			};
		}
	};
}
