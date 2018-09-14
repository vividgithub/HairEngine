//
// Created by vivi on 2018/9/13.
//

#pragma once

#include <VPly/vply.h>

#include <array>
#include <algorithm>
#include "solver.h"
#include "visualizer.h"

namespace HairEngine {

	/**
	 * An interface for the SDF collision. TRY to update the mesh before every "solve" is called. Currently, we only
	 * support triangle mesh.
	 */
	struct SDFCollisionMeshInterface {
		/**
		 * Get number of point in the mesh
		 * @return The number of point
		 */
		virtual int getPointCount() const = 0;

		/**
		 * Get number of triangle primitive in the mesh
		 * @return The number of triangle primtives
		 */
		virtual int getPrimitiveCount() const = 0;

		/**
		 * Get the 3 point index for every triangle primitives
		 * @param primIdx The primitive index
		 * @return A array<int, 3> indicating the 3 point index
		 */
		virtual std::array<int, 3> getPrimitivePointIndices(int primIdx) const = 0;

		/**
		 * Get the point position for specified index
		 * @param pointIdx
		 * @return The point position
		 */
		virtual Eigen::Vector3f getPointPosition(int pointIdx) const = 0;
	};

	struct SDFCollisionConfiguration {
		std::array<int, 3> resolution; ///< The resolution for each axes
		float extend; ///< The bounding box extend to build the SDF grid
		int narrowBandMargin; ///< The number of point margin for the narrow band width for the 0 isocontour
	};

	/**
	 * Signed distance field collision solver. It is used for resolving hair/solid collision during simulation.
	 * The solver accepts an SDFMeshInterface as an mesh (skinned mesh) input. The solve uses the mesh to generate
	 * an signed distance field and use that field to resolve hair/solid collision.
	 */
	class SDFCollisionSolver: public Solver {

		friend class SDFCollisionVisualizer;

	HairEngine_Public:

		struct SDFGridStruct {
			float dist; ///< The sdf distance
			int primIdx; ///< The primitive index that give the value
		};

		struct SDFGridCell {
			std::array<SDFGridStruct *, 8> nodes; ///< The nodes in the cell
		};

		using MeshInterface = SDFCollisionMeshInterface;
		using Configuration = SDFCollisionConfiguration;

		static const constexpr float DISTANCE_INVALID = 1e30f;

		/**
		 * Constructor
		 * @param conf The signed distance field configuration
		 * @param mesh The input mesh interface
		 */
		SDFCollisionSolver(const Configuration & conf, MeshInterface *mesh): mesh(mesh), conf(conf) {}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {
			// Setup the space for the prePoses and poses
			prePoses.resize(static_cast<size_t>(npoint()));
			for (int i = 0; i < npoint(); ++i)
				prePoses[i] = pos(i);

			// Allocating space for grid
			nz = conf.resolution[2];
			nyz = conf.resolution[1] * nz;
			nxyz = conf.resolution[0] * nyz;

			HairEngine_AllocatorAllocate(grid, static_cast<size_t>(nxyz));
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {

			std::cout << "Building sdf..." << std::endl;

			// Get the bounding box
			bbox = Eigen::AlignedBox3f(pos(0));
			for (int i = 1; i < npoint(); ++i) {
				bbox.extend(pos(i));
			}

			// Scale the bounding box
			bbox = MathUtility::scaleBoundingBox(bbox, 1.0f + conf.extend);

			// Update the d value
			d = bbox.diagonal().cwiseQuotient(
					Eigen::Vector3f(conf.resolution[0] - 1, conf.resolution[1] - 1, conf.resolution[2] - 1));
			dInv = d.cwiseInverse();

			// Clear the grid
			std::fill(grid, grid + nxyz, SDFGridStruct { DISTANCE_INVALID, -1 });

			const Eigen::Vector3i margin3i = Eigen::Vector3i::Ones() * conf.narrowBandMargin;

			// Recompute the signed distance field
			for (int primIdx = 0; primIdx < nprim(); ++primIdx) {
				auto indices = primIndices(primIdx);
				std::array<Eigen::Vector3f, 3> p { pos(indices[0]), pos(indices[1]), pos(indices[2]) };

				Eigen::AlignedBox3f primBound(p[0]);
				primBound.extend(p[1]);
				primBound.extend(p[2]);

				// FIXME: More efficient
				Eigen::Vector3i minIndex = (primBound.min() - bbox.min()).cwiseProduct(dInv).cast<int>() - margin3i;
				Eigen::Vector3i maxIndex = ((primBound.max() - bbox.min()).cwiseProduct(dInv) + Eigen::Vector3f::Ones()).cast<int>() + margin3i;

				minIndex = minIndex.cwiseMax(Eigen::Vector3i::Zero());
				maxIndex = maxIndex.cwiseMin(Eigen::Vector3i(conf.resolution[0], conf.resolution[1], conf.resolution[2]));

				// Iterate and compute the signed distance for those grid node
				for (int ix = minIndex.x(); ix <= maxIndex.x(); ++ix)
					for (int iy = minIndex.y(); iy <= maxIndex.y(); ++iy)
						for (int iz = minIndex.z(); iz <= maxIndex.z(); ++iz) {
							auto & node = grid[ix * nyz + iy * nz + iz];
							Eigen::Vector3f nodePos = bbox.min() + Eigen::Vector3f(ix, iy, iz).cwiseProduct(d);

							float primDist = MathUtility::pointToTriangleSignedDistance(nodePos, p[0], p[1], p[2]);
							if (std::abs(node.dist) > std::abs(primDist)) {
								node.dist = primDist;
								node.primIdx = primIdx;
							}
						}
			}

			//TODO: Collision fix

			// Store the position into prePoses
			for (int i = 0; i < npoint(); ++i)
				prePoses[i] = pos(i);
		}

		void tearDown() override {
			HairEngine_AllocatorDeallocate(grid, static_cast<size_t>(nxyz));
		}

		/**
		 * Transform a index (vector3) to a offset index for the grid
		 * @param index The 3 indices for x, y and z axes
		 * @return The offset for the grid
		 */
		int offset(const Eigen::Vector3i & idx) {
			return idx(0) * nyz + idx(1) * nz + idx(2);
		}

	HairEngine_Protected:

		MeshInterface *mesh; ///< The mesh interface pointer
		Configuration conf; ///< The configuration

		/// the previous point positions after the "solve" called
		/// used to compute the velocity
		std::vector<Eigen::Vector3f> prePoses;

		Eigen::AlignedBox3f bbox; ///< The current bounding box

		int npoint() const { return mesh->getPointCount(); }
		int nprim() const { return mesh->getPrimitiveCount(); }
		std::array<int, 3> primIndices(int primIdx) const { return mesh->getPrimitivePointIndices(primIdx); }
		Eigen::Vector3f pos(int pointIdx) const { return mesh->getPointPosition(pointIdx); }

		SDFGridStruct *grid; ///< The discrete grid
		int nxyz; ///< The total size of the grid
		int nyz; ///< The offset for the first dimension
		int nz; ///< The offset for the second dimension

		Eigen::Vector3f d; ///< The distance (step) for grid in each dimensions
		Eigen::Vector3f dInv; ///< The element wise inverse of d
	};

	/**
	 * A simple visualizer for the SDF collisions. Every SDF field generated by SDFCollisionSolver, it will write
	 * two kinds of VPly points. The first kind of points indicating the node whose signed distance field is not 0.
	 * And the second point shows the zero isocontour of the SDF file.
	 */
	class SDFCollisionVisualizer: public Visualizer {

	HairEngine_Public:

		SDFCollisionVisualizer(const std::string &directory, const std::string &filenameTemplate, float timestep, SDFCollisionSolver * sdfSolver)
				: Visualizer(directory, filenameTemplate, timestep), sdfSolver(sdfSolver) {}

		void visualize(std::ostream &os, Hair &hair, const IntegrationInfo &info) override {
			if (!sdfSolver)
				return;

			std::cout << "[SDFCollisionVisualizer] Write vply..." << std::endl;

			const auto & resolution =sdfSolver->conf.resolution;

			// Write the cell with 0 contour
			for (int ix = 0; ix < resolution[0] - 1; ++ix)
				for (int iy = 0; iy < resolution[1] - 1; ++iy)
					for (int iz = 0; iz < resolution[2] - 1; ++iz) {

						const auto index3 = Eigen::Vector3i(ix, iy, iz);

						SDFCollisionSolver::SDFGridStruct *origin = sdfSolver->grid + sdfSolver->offset(index3);
						std::array<SDFCollisionSolver::SDFGridStruct *, 8> nodes = {
								origin,
								origin + 1,
								origin + sdfSolver->nz,
								origin + sdfSolver->nz + 1,
								origin + sdfSolver->nyz,
								origin + sdfSolver->nyz + 1,
								origin + sdfSolver->nyz + sdfSolver->nz,
								origin + sdfSolver->nyz + sdfSolver->nz + 1
						};

						// Continue if all the nodes are invalid
						if (std::any_of(nodes.begin(), nodes.end(), [] (SDFCollisionSolver::SDFGridStruct * node) -> bool { return node->dist == SDFCollisionSolver::DISTANCE_INVALID; }))
							continue;

						// Type Flags:
						// 0: Normal cell
						// 1: Boundary cell
						// 2: Zero contour cell
						int typeFlag = 0;
						const Eigen::Vector3f cellCenter = sdfSolver->bbox.min()
						                                   + (index3.cast<float>() + Eigen::Vector3f(0.5f, 0.5f, 0.5f)).cwiseProduct(sdfSolver->d);

						if (std::any_of(nodes.begin() + 1, nodes.end(), [origin] (SDFCollisionSolver::SDFGridStruct *node) -> bool { return (node->dist >= 0.0f && origin->dist <= 0.0f) || (node->dist <= 0.0f && origin->dist >= 0.0f); })) {
							typeFlag = 1;
						}

						VPly::AttributedLineStrip l(2, VPly::VPlyIntAttr("type", typeFlag));

						const auto & d = sdfSolver->d;

						Eigen::Vector3f originPos = index3.cast<float>().cwiseProduct(d) + sdfSolver->bbox.min();
						std::array<Eigen::Vector3f, 8> poses = {
								originPos,
								originPos + Eigen::Vector3f(0.0, 0.0, d.z()),
								originPos + Eigen::Vector3f(0.0, d.y(), 0.0f),
								originPos + Eigen::Vector3f(0.0, d.y(), d.z()),
								originPos + Eigen::Vector3f(d.x(), 0.0f, 0.0f),
								originPos + Eigen::Vector3f(d.x(), 0.0, d.z()),
								originPos + Eigen::Vector3f(d.x(), d.y(), 0.0f),
								originPos + Eigen::Vector3f(d.x(), d.y(), d.z())
						};

						// 0 -> 1 -> 3 -> 2 -> 0 -> 4 -> 6 -> 2 -> 6 -> 7 -> 5 -> 4 -> 5 -> 1 -> 3 -> 7
						std::array<int, 16> drawIndices = { 0, 1, 3, 2, 0, 4, 6, 2, 6, 7, 5, 4, 5, 1, 3, 7 };
						for (auto drawIndex : drawIndices) {
							l.addPoint(EigenUtility::toVPlyVector3f(poses[drawIndex]), VPly::VPlyFloatAttr("dist", nodes[drawIndex]->dist), VPly::VPlyIntAttr("pid", nodes[drawIndex]->primIdx));
						}

						l.stream(os);
					};
		}

	HairEngine_Protected:
		SDFCollisionSolver *sdfSolver;
	};
}
