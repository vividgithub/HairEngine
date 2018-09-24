//
// Created by vivi on 2018/9/13.
//

#pragma once

#include <VPly/vply.h>

#ifdef HAIRENGINE_ENABLE_CUDA
#include "../util/cudautil.h"
#include "cuda_memory_converter.h"
#endif

#include <array>
#include <algorithm>
#include "solver.h"
#include "visualizer.h"
#include "integrator.h"

void SDFCollisionSolver_cudaComputeVelocities(const float3 *prePoses,
                                                    const float3 *poses, const int3 *indices,
                                                    float3 *outVel, float tInv, int nprim, int nblock, int nthread);

void SDFCollisionSolver_cudaComputeSDFGrid(const float3 *poses, const int3 *indices, unsigned long long * outGrid,
                                           int npoint, int nprim, int3 n, int margin, float3 origin, float3 d, int nblock, int nthread);

void SDFCollisionSolver_cudaResolveCollision(float3 *parPoses, float3 *parVels, const unsigned char *parLocalIndices, const float3 * vels,
                                             const unsigned long long *grid, int npar, float3 origin, float3 d, int3 n, float contour, float fraction, bool changeHairRoot, int nblock, int nthread);

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

		// Collision
		/// Instead of pushing it to the 0 iso contour, we support to push it a slightly far away. The
		/// absolute contour to push to is defined by "relativeContour * the_diagnoal_length_of_the_rest_position".
		float relativeContour;

		/// The friction coefficient of the SDF object
		float fraction;

		/// To ensure no-relative position flip, instead of pushing all the particle into the same relative contour,
		/// we support to push the particle further away if it is more "inside" the SDFCollision object. The push
		/// distance can be expressed as "(absolute_contour - signed_distance_of_the_particle) * (1 + degenerationFactor)"
		float degenerationFactor;

		/// Whether to change hair root
		bool changeHairRoot;

		/// The thread size for each thread block when using cuda for computation
		int cudaThreadSize = 256;
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
			Eigen::Vector3f vel; ///< The projection veloicties
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
		SDFCollisionSolver(
				const Configuration & conf,
				MeshInterface *mesh
		):
			mesh(mesh),
			conf(conf),
			nx(conf.resolution[0]),
			ny(conf.resolution[1]),
			nz(conf.resolution[2]),
			nyz(ny * nz),
			nxyz(nx * nyz),
			n(nx, ny, nz),
			_1PlusDegenerationFactor(1.0f + conf.degenerationFactor) {

			// Pre-allocating the space for grid
#ifndef HAIRENGINE_ENABLE_CUDA
			HairEngine_AllocatorAllocate(grid, static_cast<size_t>(nxyz));
			gridLocks = new ParallismUtility::Spinlock[static_cast<size_t>(nxyz)];
#else
			HairEngine_CudaAllocatorAllocate(grid, nxyz);
#endif
		}

		void setup(const Hair &hair, const Eigen::Affine3f &currentTransform) override {

#ifndef HAIRENGINE_ENABLE_CUDA
			// Setup the space for the prePoses and poses
			prePoses.resize(static_cast<size_t>(npoint()));
			vels.resize(static_cast<size_t>(npoint()));
			for (int i = 0; i < npoint(); ++i)
				prePoses[i] = pos(i);
#else
			// Find the last CudaMemoryConverter
			cmc = integrator->rfindSolver<CudaMemoryConverter>(0, solverIndex);
			if (!cmc)
				throw HairEngineInitializationException("The CudaMemoryConverter is not initialized before the SDFCollisionSolver");
			if (cmc->parPoses == nullptr || cmc->parVels == nullptr || cmc->parLocalIndices == nullptr)
				throw HairEngineInitializationException("The CudaMemoryConverter has not copied particle position or velocities or local indices into the device buffer");

			HairEngine_CudaAllocatorAllocate(prePoses, npoint());
			HairEngine_CudaAllocatorAllocate(poses, npoint());

			HairEngine_CudaAllocatorAllocate(vels, nprim());
			HairEngine_CudaAllocatorAllocate(indices, nprim());
			posesHost = new float3[npoint()];

			copyPosFromHostToDevice(prePoses);

			// Copy the indices
			int3 *indicesHost;
			HairEngine_AllocatorAllocate(indicesHost, nprim());
			for (int i = 0; i < nprim(); ++i) {
				auto meshPrimIndices = primIndices(i);
				indicesHost[i].x = meshPrimIndices[0];
				indicesHost[i].y = meshPrimIndices[1];
				indicesHost[i].z = meshPrimIndices[2];
			}
			cudaMemcpy(indices, indicesHost, sizeof(int3) * static_cast<size_t>(nprim()), cudaMemcpyHostToDevice);
			HairEngine_AllocatorDeallocate(indicesHost, nprim());
#endif

			// Compute the absContour
			Eigen::AlignedBox3f restBBox(pos(0));
			for (int i = 1; i < npoint(); ++i)
				restBBox.extend(pos(i));

			absContour = restBBox.diagonal().norm() * conf.relativeContour;
		}

		void solve(Hair &hair, const IntegrationInfo &info) override {

			std::cout << "Building sdf..." << std::endl;

			computePointVelocities(info);
			buildSDFField();
			resolveCollision(hair, info);
			storePreviousPositions();
		}

	HairEngine_Protected:

		void tearDown() override {
#ifndef HAIRENGINE_ENABLE_CUDA
			HairEngine_AllocatorDeallocate(grid, static_cast<size_t>(nxyz));
			delete [] gridLocks;
#else
			cudaFree(grid);
			cudaFree(prePoses);
			cudaFree(poses);
			cudaFree(vels);
			cudaFree(indices);

			delete[] posesHost;
#endif
		}

		/**
		 * Transform a index (vector3) to a offset index for the grid
		 * @param index The 3 indices for x, y and z axes
		 * @return The offset for the grid
		 */
		int offset(const Eigen::Vector3i & idx) {
			return idx(0) * nyz + idx(1) * nz + idx(2);
		}

		/**
		 * Get the node index for each axis from the position, suppose the position is surrounded by the node with index
		 * (x, y, z) and (x + 1, y + 1, z + 1), the funciton will return (x, y, z) instead of (x + 1, y + 1, z + 1_
		 * @param pos The position
		 * @return A Eigen::Vector3i indicating the index for each axis
		 */
		Eigen::Vector3i getNodeIndex(const Eigen::Vector3f & pos) {
			return (pos - bbox.min()).cwiseProduct(dInv).cast<int>();
		}

//		/**
//		 * Get the surrounding voxel (cell) from a given position
//		 * @param pos The position
//		 * @return The cell which the position is inside, ordered by the incresment of z, then y, then x
//		 */
//		SDFGridCell getCell(const Eigen::Vector3f & pos) {
//
//			SDFGridStruct *origin;
//			origin = grid + offset(getNodeIndex(pos));
//
//			return {
//				origin, origin + 1, origin + nz, origin + nz + 1,
//				origin + nyz, origin + nyz + 1, origin + nyz + nz, origin + nyz + nz + 1
//			};
//		}

#ifndef HAIRENGINE_ENABLE_CUDA
		/**
		 * Get the SDF object velocity from the primitive index and the uv coordinate of that primitive.
		 * Make sure to call the method after the "vels" have been computed correctly.
		 * @param primIdx The primitive index
		 * @param uv The LOCAL uv coordinate of the primitive
		 * @return The interpolated velocity
		 */
		Eigen::Vector3f getSDFVelocity(int primIdx, const Eigen::Vector2f & uv) {
			auto indices = mesh->getPrimitivePointIndices(primIdx);
			return (vels[indices[0]] - vels[indices[2]]) * uv(0)
					+ (vels[indices[1]] - vels[indices[2]]) * uv(1)
					+ vels[indices[2]];
		}
#endif


		MeshInterface *mesh; ///< The mesh interface pointer
		Configuration conf; ///< The configuration

#ifndef HAIRENGINE_ENABLE_CUDA
		/// the previous point positions after the "solve" called
		/// used to compute the velocity
		std::vector<Eigen::Vector3f> prePoses;
		std::vector<Eigen::Vector3f> vels; ///< The velocity buffers

		SDFGridStruct *grid; ///< The discrete grid
		ParallismUtility::Spinlock *gridLocks; ///< The spin lock for each grid node
#else
		float3 *prePoses, *poses, *vels; ///< The mesh point positions (device) and velocities(device)
		int3 *indices; ///< The prim indices (device)
		float3 *posesHost; ///< The host point positions

		/// We pack the minimum signed distance, the prim index and the uv coordinate into a single 64 bit
		/// unsigned long long type so that we can use "atomicMin" in cuda for the signed distance. The 64bit
		/// is composed of a 32 bit signed distance, a 16 bit primIdx (so the total primitive should not excceed
		/// 65535) and 2 * 8bit u, v coordinate.
		unsigned long long *grid;

		/// We use CudaMemoryConverter to pass the particle positions that should be modified
		CudaMemoryConverter *cmc;
#endif

		float absContour; ///< The absolute target isocontour
		float _1PlusDegenerationFactor; ///< 1.0f + conf.degnerationFactor;

		int nx, ny, nz; ///< The resolution for each dimension
		int nyz; ///< The offset for the first dimension
		int nxyz; ///< The total size of the grid
		Eigen::Vector3i n; ///< Equals to resolution, in Vector3i

		Eigen::Vector3f d; ///< The distance (step) for grid in each dimensions
		Eigen::Vector3f dInv; ///< The element wise inverse of d
		Eigen::AlignedBox3f bbox; ///< The current bounding box

//		struct CollisionInfo {
//			Eigen::Vector3f prevPos, pos;
//			Eigen::Vector3f gradient;
//			float signedDistance;
//		};

//		std::vector<CollisionInfo> infos;

		int npoint() const { return mesh->getPointCount(); }
		int nprim() const { return mesh->getPrimitiveCount(); }
		std::array<int, 3> primIndices(int primIdx) const { return mesh->getPrimitivePointIndices(primIdx); }
		Eigen::Vector3f pos(int pointIdx) const { return mesh->getPointPosition(pointIdx); }

#ifdef HAIRENGINE_ENABLE_CUDA
		void cudaComputeSDFGrid() {
			const int & nthread = conf.cudaThreadSize;
			int nblock = (nprim() + nthread - 1) / nthread;

			SDFCollisionSolver_cudaComputeSDFGrid(
					poses,
					indices,
					grid,
					npoint(),
					nprim(),
					{n.x(), n.y(), n.z() },
					conf.narrowBandMargin,
					EigenUtility::toFloat3(bbox.min()),
					EigenUtility::toFloat3(d),
					nblock,
					nthread
			);
		}

		void cudaComputeVelocities(float t) {
			float tInv = 1.0f / t;

			const int & nthread = conf.cudaThreadSize;
			int nblock = (nprim() + nthread - 1) / nthread;

			SDFCollisionSolver_cudaComputeVelocities(prePoses, poses, indices, vels, tInv, nprim(), nblock, nthread);
		}

		void cudaResolveCollision() {
			const int & nthred = conf.cudaThreadSize;
			int nblock = (hair->nparticle + nthred - 1) / nthred;

			SDFCollisionSolver_cudaResolveCollision(
					cmc->parPoses,
					cmc->parVels,
					cmc->parLocalIndices,
					vels,
					grid,
					hair->nparticle,
					EigenUtility::toFloat3(bbox.min()),
					EigenUtility::toFloat3(d),
					{n.x(), n.y(), n.z() },
					absContour,
					conf.fraction,
					conf.changeHairRoot,
					nblock,
					nthred
			);
		}

		/**
		 * Copy the pos from the mesh interface to the device float3 array
		 * @param devicePtr The float3 device pointer which points to a continous memory
		 */
		void copyPosFromHostToDevice(float3 *devicePtr) {
			for (int i = 0; i < npoint(); ++i)
				posesHost[i] = EigenUtility::toFloat3(pos(i));
			CudaUtility::copyFromHostToDevice(devicePtr, posesHost, npoint());
		}
#endif

		void buildSDFField() {
			// Get the bounding box
			bbox = Eigen::AlignedBox3f(pos(0));
			for (int i = 1; i < npoint(); ++i) {
				bbox.extend(pos(i));
			}

			// Scale the bounding box
			bbox = MathUtility::scaleBoundingBox(bbox, 1.0f + conf.extend);

			// Update the d value
			d = bbox.diagonal().cwiseQuotient((n - Eigen::Vector3i::Ones()).cast<float>());
			dInv = d.cwiseInverse();

#ifndef HAIRENGINE_ENABLE_CUDA
			// Clear the grid
			std::fill(grid, grid + nxyz, SDFGridStruct { DISTANCE_INVALID, Eigen::Vector3f::Zero()});

			const Eigen::Vector3i margin3i = Eigen::Vector3i::Ones() * conf.narrowBandMargin;

			// Recompute the signed distance field
			ParallismUtility::parallelFor(0, nprim(), [this, &margin3i] (int primIdx){
				auto indices = primIndices(primIdx);
				std::array<Eigen::Vector3f, 3> p { pos(indices[0]), pos(indices[1]), pos(indices[2]) };

				Eigen::AlignedBox3f primBound(p[0]);
				primBound.extend(p[1]);
				primBound.extend(p[2]);

				Eigen::Vector3i minIndex = (getNodeIndex(primBound.min()) - margin3i).cwiseMax(Eigen::Vector3i::Zero());
				// Ceiling the maxIndex, add Vector3i::Ones() to it so that there's at least a voxel between the minIndex and maxIndex
				Eigen::Vector3i maxIndex = (getNodeIndex(primBound.max()) + margin3i + Eigen::Vector3i::Ones()).cwiseMin(n);

				// Iterate and compute the signed distance for those grid node
				for (int ix = minIndex.x(); ix <= maxIndex.x(); ++ix)
					for (int iy = minIndex.y(); iy <= maxIndex.y(); ++iy)
						for (int iz = minIndex.z(); iz <= maxIndex.z(); ++iz) {

							int off = offset({ix, iy, iz});

							auto & node = grid[off];
							auto & spinLock = gridLocks[off];

							Eigen::Vector3f nodePos = bbox.min() + Eigen::Vector3f(ix, iy, iz).cwiseProduct(d);

							Eigen::Vector2f uv;
							float primDist = MathUtility::pointToTriangleSignedDistance(nodePos, p[0], p[1], p[2], &uv);

							spinLock.lock();
							if (std::abs(node.dist) > std::abs(primDist)) {
								node.dist = primDist;
								node.vel = getSDFVelocity(primIdx, uv);
							}
							spinLock.unlock();
						}
			});
#else
			// Clear the grid
			cudaMemset(grid, 0xff, sizeof(unsigned long long) * nxyz);
			cudaComputeSDFGrid();
#endif
		}

		void computePointVelocities(const IntegrationInfo & info) {
#ifndef HAIRENGINE_ENABLE_CUDA
			// Compute the particle velocities
			float tInv = 1.0f / info.t;

			ParallismUtility::parallelFor(0, npoint(), [this, tInv] (int i) {
				vels[i] = (pos(i) - prePoses[i]) * tInv;
			});
#else
			// Copy the position to poses
			copyPosFromHostToDevice(poses);
			cudaComputeVelocities(info.t);
#endif
		}

		void resolveCollision(Hair &hair, const IntegrationInfo &info) {
#ifndef HAIRENGINE_ENABLE_CUDA
			//infos.clear();

			// Resolve collision
			mapParticle(true, [this] (Hair::Particle *par) {
				// Not in the extended bounding box
				if (!bbox.contains(par->pos))
					return;
				else if (par->localIndex == 0 && !conf.changeHairRoot)
					return;

				// Query the surrounding cell
				Eigen::Vector3f index3f = (par->pos - bbox.min()).cwiseProduct(dInv);
				Eigen::Vector3i index3 = index3f.cast<int>().cwiseMin(n - Eigen::Vector3i::Ones());

				// Query the surrounded nodes
				SDFGridStruct *origin = grid + offset(index3);
				std::array<SDFGridStruct *, 8> nodes = {
						origin, origin + 1, origin + nz, origin + nz + 1,
						origin + nyz, origin + nyz + 1, origin + nyz + nz, origin + nyz + nz + 1
				};

				// Invalid cell, ignore it
				if (std::any_of(nodes.begin(), nodes.end(),
				                [this] (SDFGridStruct *node) -> bool { return node->dist == SDFCollisionSolver::DISTANCE_INVALID; }))
					return;

				// All are above the isocontour, ignore it
				else if (std::all_of(nodes.begin(), nodes.end(),
						[this] (SDFGridStruct *node) -> bool { return node->dist > absContour; }))
					return;

				// Get the interpolation weights by tri-linear interpolation
				Eigen::Vector3f t = index3f - index3.cast<float>();
				float chunkx[2] {1 - t.x(), t.x()};
				float chunky[2] {1 - t.y(), t.y()};
				float chunkz[2] {1 - t.z(), t.z()};

				//Compute the distance and the gradient
				float signedDist = 0.0f;

				for (int i = 0; i < 8; ++i) {
					int boolx = (i >> 2), booly = (i >> 1) & 1, boolz = i & 1;
					signedDist += nodes[i]->dist * chunkx[boolx] * chunky[booly] * chunkz[boolz];
				}

				if (signedDist > absContour)
					return;

				// Compute the gradient and velocity of the object
				Eigen::Vector3f gradient = Eigen::Vector3f::Zero();
				Eigen::Vector3f v = Eigen::Vector3f::Zero(); // Object velocity

				for (int i = 0; i < 8; ++i) {
					int boolx = (i >> 2), booly = (i >> 1) & 1, boolz = i & 1;
					const auto & node = nodes[i];

					v += node->vel * (chunkx[boolx] * chunky[booly] * chunkz[boolz]);

					gradient(0) += (boolx ? 1.0f : -1.0f) * node->dist * chunky[booly] * chunkz[boolz];
					gradient(1) += (booly ? 1.0f : -1.0f) * node->dist * chunkx[boolx] * chunkz[boolz];
					gradient(2) += (boolz ? 1.0f : -1.0f) * node->dist * chunkx[boolx] * chunky[booly];
				}

				gradient = gradient.cwiseProduct(dInv);

				gradient.normalize();

				// Compute the fixed position and velocity
				auto & vp = par->vel; // Particle velocity
				Eigen::Vector3f vpt, vpn, vt, vn; // Projection and tangent object and particle velocity

				MathUtility::projection(vp, gradient, vpn, vpt);
				MathUtility::projection(v, gradient, vn, vt);

				//Update the tangent velocity
				Eigen::Vector3f vrelt = vpt - vt;
				vrelt = std::max<float>(0.0f, 1.0f - conf.fraction * (vpn - vn).norm() / vrelt.norm()) * vrelt;
				vpt = vt + vrelt;

				//Update the normal velocity
				vpn = vn;

				par->vel = vpt + vpn;
//				infos.emplace_back();
//				auto & info = infos.back();
//				info.prevPos = par->pos;
//				info.gradient = gradient;
//				info.signedDistance = signedDist;
//
				par->pos += ((absContour - signedDist)) * gradient;
//				par->vel = Eigen::Vector3f::Zero();
//
//				info.pos = par->pos;

 			});

//			std::cout << infos.size() << std::endl;
#else
			cudaResolveCollision();
#endif
		}

		void storePreviousPositions() {
#ifndef HAIRENGINE_ENABLE_CUDA
			ParallismUtility::parallelFor(0, npoint(), [this](int i){
				prePoses[i] = pos(i);
			});

#else
			std::swap(prePoses, poses);
#endif
		}
	};

	/**
	 * A simple visualizer for the SDF collisions. Every SDF field generated by SDFCollisionSolver, it will write
	 * two kinds of VPly points. The first kind of points indicating the node whose signed distance field is not 0.
	 * And the second point shows the zero isocontour of the SDF file.
	 */
	class SDFCollisionVisualizer: public Visualizer {

	HairEngine_Public:

		SDFCollisionVisualizer(const std::string &directory, const std::string &filenameTemplate, float timestep, SDFCollisionSolver * sdfSolver)
				: Visualizer(directory, filenameTemplate, timestep), sdfSolver(sdfSolver) {
#ifdef HAIRENGINE_ENABLE_CUDA
			if (sdfSolver) {
				posesHost = new float3[sdfSolver->npoint()];
				gridHost = new unsigned long long[sdfSolver->nxyz];
				velsHost = new float3[sdfSolver->nprim()];
			}
#endif
		}

		void tearDown() override {
			if (sdfSolver) {
				delete [] posesHost;
				delete[] gridHost;
				delete[] velsHost;
			}
		}

		void visualize(std::ostream &os, Hair &hair, const IntegrationInfo &info) override {
			if (!sdfSolver)
				return;

			std::cout << "[SDFCollisionVisualizer] Write vply..." << std::endl;

			const auto & resolution =sdfSolver->conf.resolution;

#ifndef HAIRENGINE_ENABLE_CUDA
#else
			// Copy the data to host
			CudaUtility::copyFromDeviceToHost(posesHost, sdfSolver->poses, sdfSolver->npoint());
			CudaUtility::copyFromDeviceToHost(velsHost, sdfSolver->vels, sdfSolver->nprim());
			CudaUtility::copyFromDeviceToHost(gridHost, sdfSolver->grid, sdfSolver->nxyz);
#endif

			// Write the mesh surface


#ifndef HAIRENGINE_ENABLE_CUDA
			for (int i = 0; i < sdfSolver->npoint(); ++i) {
				VPly::VPlyVector3f writeVel = EigenUtility::toVPlyVector3f(sdfSolver->vels[i]);
				VPly::writePoint(
						os, EigenUtility::toVPlyVector3f(sdfSolver->pos(i)),
						VPly::VPlyIntAttr("type", 1),
						VPly::VPlyVector3fAttr("v", writeVel)
				);
			}
#else
			for (int i = 0; i < sdfSolver->nprim(); ++i) {
				auto pids = sdfSolver->primIndices(i);
				float3 vel = velsHost[i];

				VPly::AttributedLineStrip l(
						2, VPly::VPlyVector3fAttr("vel", { vel.x, vel.y, vel.z })
				);

				for (auto pid : pids) {
					auto pos = posesHost[pid];
					l.addPoint({ pos.x, pos.y, pos.z }, VPly::VPlyIntAttr("type", 1));
				}

				l.stream(os);
			}
#endif


		// Write the cell with 0 and abs contour
			for (int ix = 0; ix < resolution[0] - 1; ++ix)
				for (int iy = 0; iy < resolution[1] - 1; ++iy)
					for (int iz = 0; iz < resolution[2] - 1; ++iz) {

						Eigen::Vector3i index3 = Eigen::Vector3i(ix, iy, iz);
						const Eigen::Vector3f pos = sdfSolver->bbox.min() + index3.cast<float>().cwiseProduct(sdfSolver->d);

#ifndef HAIRENGINE_ENABLE_CUDA
						auto & node = sdfSolver->grid[sdfSolver->offset(Eigen::Vector3i(ix, iy, iz))];

						if (node.dist == SDFCollisionSolver::DISTANCE_INVALID)
							continue;


						VPly::writePoint(
								os, EigenUtility::toVPlyVector3f(pos),
								VPly::VPlyVector3iAttr("i3", EigenUtility::toVPlyVector3i(index3)),
								VPly::VPlyFloatAttr("d", node.dist),
								VPly::VPlyVector3fAttr("v", EigenUtility::toVPlyVector3f(node.vel))
						);
#else
						unsigned long long node = gridHost[sdfSolver->offset(index3)];
						uint32_t *nodeAddr = reinterpret_cast<uint32_t*>(&node);

						if (node == 0xffffffffffffffff)
							continue;

						// Unpack the dist, primIdx, and the u, v
						uint32_t flipDist = nodeAddr[1]; // Little endian
						flipDist = (flipDist >> 1) | (flipDist << 31);
						float dist = reinterpret_cast<float*>(&flipDist)[0];

						uint32_t primIdx = nodeAddr[0];

						auto vel = velsHost[primIdx];

						VPly::writePoint(
								os, EigenUtility::toVPlyVector3f(pos),
								VPly::VPlyVector3iAttr("i3", EigenUtility::toVPlyVector3i(index3)),
								VPly::VPlyFloatAttr("d", dist),
								VPly::VPlyVector3fAttr("v", { vel.x, vel.y, vel.z })
						);
#endif
					}

		}

	HairEngine_Protected:
		SDFCollisionSolver *sdfSolver;

#ifdef HAIRENGINE_ENABLE_CUDA
		unsigned long long *gridHost = nullptr; ///< The grid data (copy to host)
		float3 *velsHost = nullptr; ///< The velocities (copy to host)
		float3 *posesHost = nullptr; ///< The poses (point to host)
#endif
	};
}

