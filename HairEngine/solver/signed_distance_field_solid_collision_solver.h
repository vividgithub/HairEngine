#pragma once

#include <array>
#include <cstdio>

#include "solid_collision_solver_base.h"
#include "../util/stringutil.h"
#include "HairEngine/HairEngine/util/mathutil.h"

namespace HairEngine {
	/*
	* The data store in the SDF Collider vertex
	*/
	struct SDFColliderVertex {

		typedef SDFColliderVertex *Ptr;

		float distance; //the distance to the surface of the SDFCollider
	};

	/*
	* A cell is represented by 8 surrounding vertexes
	*/
	struct SDFColliderCell {
		typedef SDFColliderCell *Ptr;

		/*
		* The pointers are ordered by:
		*     v[0] ---> The lowest vertex in the cell
		*     v[1] ---> v[0] move along z axis
		*     v[2] ---> v[0] move along y axis
		*     v[3] ---> v[0] move along y and z axis
		*     v[4] ---> v[0] move along x axis
		*     v[5] ---> v[4] move along z axis
		*     v[6] ---> v[4] move along y axis
		*     v[7] ---> v[4] move along y and z axis
		* In short, if we see the index in binary format [ax, ay, az], when a? = 0 means not moving in that axis from v[0]
		* , otherwise move in that axis
		*
		*/
		std::array<SDFColliderVertex::Ptr, 8> v;

		SDFColliderCell() = default;
		SDFColliderCell(const SDFColliderCell &other) = default;
	};

	/**
	 * SignedDistanceFieldSolidCollisionSolver (signed distance field collider) is a collision solver which the distance information
	 * has been precomputed in a grid based voxel lattice. It supports a constant time distance and gradient
	 * query method. The SDFCollider can be initialized from a storage file .sdf (binary) and .sdf2 (binary)
	 *
	 * The .sdf, .sdf2 file format:
	 *     <nx> <ny> <nz> ===> [int32_t]
	 *     <originX> <originY> <originZ> ===> [float]
	 *     <dx> for sdf (or <dx> <dy> <dz> for .sdf2) ===> [float]
	 *     <value1> <value2> <value3> ... ===> [float]
	 *
	 *	ni, nj, nk are the integers that tells how many vertexes in the lattice of x, y and z coordinates,
	 *  originX, originY, originZ tells the start origin (which is the lowest point), dx, dy, dz tells the size of
	 *  cell(8 nearby vertex forms a cell) in x, y, z coordinates.
	 *  And value1... stores the distance value of the each points
	 */
	class SignedDistanceFieldSolidCollisionSolver: public SolidCollisionSolverBase {

	HairEngine_Public:
		using Cell = SDFColliderCell;
		using Vertex = SDFColliderVertex;

		constexpr static const float SDF_INFINITY = 1e30f;

	HairEngine_Public:
		//it could be initialized with .sdf file format and .sdf2 file format
		SignedDistanceFieldSolidCollisionSolver(const std::string & filePath, const Eigen::Affine3f & transform, 
			const Eigen::Affine3f & previousTransform, const float t, const Configuration & conf):
			SolidCollisionSolverBase(transform, previousTransform, t, conf) {
			init(filePath);
		}

		SignedDistanceFieldSolidCollisionSolver(const std::string & filePath, const Eigen::Affine3f & transform, const Configuration & conf) :
			SolidCollisionSolverBase(transform, conf) {
			init(filePath);
		}

		void tearDown() override {
			if (vertexes)
				delete[] vertexes;
			SolidCollisionSolverBase::tearDown();
		}

		float modelDistance(const Eigen::Vector3f & pos, Eigen::Vector3f * outGradientPtr = nullptr) const override {
			//If not in the bounding box, we cannot get the distance
			if (!bounding.contains(pos))
				return DISTANCE_INFINITY;

			//Get the surrounding
			Eigen::Vector3f t;
			auto cell = getCell(pos, &t);
			const float & tx = t(0);
			const float & ty = t(1);
			const float & tz = t(2);

			//Those are used for prev caculation storage for computation distance and gradient using trilinear interpolation
			float chunkx[2] = {1 - tx, tx};
			float chunky[2] = {1 - ty, ty};
			float chunkz[2] = {1 - tz, tz};
			float chunkxForGradient[2] = { -1.0f / (1 - tx), 1.0f / tx};
			float chunkyForGradient[2] = { -1.0f / (1 - ty), 1.0f / ty};
			float chunkzForGradient[2] = { -1.0f / (1 - tz), 1.0f / tz};

			/*
			 * For distance, the "distance" in the voxel is interpolated with trilinear interpolation,
			 * so if the 8 neighbors are denoted by v000 (the lowest point), v001 (increased z), v010 (increased y),
			 * v011 (increased y and z), v100 (increased x), ... , v111. For the given tx, ty, tz ( 0 < x, y, z < 1), we have
			 *
			 * distanceInXYZ =
			 *         v000 * (1 - tx) * (1 - ty) * (1 - tz) +
			 *         v001 * (1 - tx) * (1 - ty) * tz +
			 *         v010 * (1 - tx) * ty * (1 - tz)
			 *         ... +
			 *         v111 * tx * ty * tz
			 *
			 *  We compute the gradient along with the computation of distance. When summing the corresponding 8 terms
			 *  (described above), if the gradient "z" term is (1 - z), it will donate an v000 * (1 - x) * (1 - y) * -1.0f to the
			 *  gradient "z" term, but the distance donation is v000 * (1 - x) * (1 - y) * (1 - z), we could use the distance donation,
			 *  multiply it by chunkZForGradient[0] and gets the result.
			 */

			//Compute the distance and the gradient
			float distance = 0.0f;
			auto gradient = Eigen::Vector3f::Zero();

			for (size_t i = 0; i < 8; ++i) {
				size_t boolx = (i >> 2), booly = (i >> 1) & 1, boolz = i & 1;

				float term = cell.v[i]->distance * (chunkx[boolx] * chunky[booly] * chunkz[boolz]);
				if (outGradientPtr != nullptr) {
					gradient(0) += (boolx ? 1.0f : -1.0f) * cell.v[i]->distance * chunky[booly] * chunkz[boolz];
					gradient(1) += (booly ? 1.0f : -1.0f) * cell.v[i]->distance * chunkx[boolx] * chunkz[boolz];
					gradient(2) += (boolz ? 1.0f : -1.0f) * cell.v[i]->distance * chunkx[boolx] * chunky[booly];
				}
				distance += term;
			}

			if (outGradientPtr != nullptr) {
				/*
				 * Since x = CONSTANT + tx * cellSize.x
				 * d(tx) / dx = 1.0f / cellSize.x
				 * By using the chain rule of derivative, we must multiply the result graident with the reciprocal of cell size
				 * in each dimension.
	            */
				*outGradientPtr = gradient.cwiseProduct(cellSizeReciprocal);
			}

			return distance;
		}


	HairEngine_Protected:

		void init(const std::string & filePath) {
			const auto & throwException = [](const std::string & message) {
				throw std::fstream::failure(message);
			};

			const auto & throwExceptionAssert = [&throwException](bool predicate, const std::string & message) {
				if (!predicate)
					throwException(message);
			};

			throwExceptionAssert(
				StringUtility::endswith(filePath, ".sdf") || StringUtility::endswith(filePath, ".sdf2"),
				"The correct file format should be endsed with .sdf or .sdf2"
			);

			std::fstream sdfFile(filePath, std::ios::in | std::ios::binary);
			throwExceptionAssert(
				sdfFile.is_open(),
				std::string("Unexpected result while opening file\"") + filePath + "\""
			);

			HairEngine_DebugIf {
				std::cout << "Begin to read sdf file of \"" << filePath << "\"" << std::endl;
			}

			nx = static_cast<int>(FileUtility::binaryReadInt32(sdfFile));
			ny = static_cast<int>(FileUtility::binaryReadInt32(sdfFile));
			nz = static_cast<int>(FileUtility::binaryReadInt32(sdfFile));
			nyz = ny * nz;
			vertexCount = nx * nyz;

			minP = FileUtility::binaryReadVector3f(sdfFile);

			if (StringUtility::endswith(filePath, ".sdf")) {
				//.sdf
				//The length of voxel in .sdf are equal in each dimensions
				const float tmp = FileUtility::binaryReadReal32(sdfFile);
				cellSize << tmp, tmp, tmp;
			}
			else {
				//.sdf2
				cellSize = FileUtility::binaryReadVector3f(sdfFile);
			}

			cellSizeReciprocal = cellSize.cwiseInverse();
			maxP = minP + Eigen::Vector3f(nx - 1, ny - 1, nz - 1).cwiseProduct(cellSize);
			bounding = Eigen::AlignedBox3f(minP, maxP);

			vertexes = new SDFColliderVertex[vertexCount];
			for (int i = 0; i < vertexCount; ++i)
				vertexes[i].distance = FileUtility::binaryReadReal32(sdfFile);

			HairEngine_DebugIf {

				std::printf("Done reading sdf file of \"%s\"\n", filePath.c_str());
				std::printf("Information about sdf file \"%s\"\n", filePath.c_str());
				std::printf("Number of vertexes: nx = %d, ny = %d, nz = %d, nyz = %d, cellSize = %s\n", (int)nx, (int)ny, (int)nz, (int)nyz, EigenUtility::toString(cellSize).c_str());
				std::printf("Bounding box: minP = %s, maxP = %s\n", EigenUtility::toString(minP).c_str(), EigenUtility::toString(maxP).c_str());
				std::printf("The distance of first %d voxel distance are: \n", std::min(vertexCount, 1000));

				for (int i = 0; i < std::min(vertexCount, 1000); ++i) {
					std::printf("%f  ", vertexes[i].distance);
					if (i % 1000 == 0)
						std::printf("\n");
				}
				std::printf("\n");

				int positiveCounter = 0, negativeCounter = 0;
				for (int i = 0; i < vertexCount; ++i) {
					if (vertexes[i].distance > 0)
						++positiveCounter;
					else
						++negativeCounter;
				}

				std::printf("Postive counter = %d, negative count = %d\n", positiveCounter, negativeCounter);
			
			}
		}

		/*
		* File data storage
		*
		* nx, ny, nz: The vertex count in the x, y, z coordinates, the total count of all vertexes is nx * ny * nz;
		* minP, maxP: The minimum bounding point and the maximum bounding point
		* vertexCount: The size of vertexes array
		* vertexes: A array, size equals to nx * ny * nz, storage the data information of each vertex, in the ascending order
		*            of x, y, z
		* cellSize: The size of a cell in x, y, z coordinates
		* cellSizeReciprocal: Equals to 1.0f / cellSize, in each coordinate, for fast computation since division
		* in computer is much slower than multiplication
		*/
		int nx, ny, nz;
		Eigen::Vector3f minP;
		Eigen::Vector3f maxP;
		Eigen::AlignedBox3f bounding;
		Eigen::Vector3f cellSize;
		Eigen::Vector3f cellSizeReciprocal; //Equals to "1.0f / cellSize", for better computation

		SDFColliderVertex *vertexes = nullptr;
		int vertexCount;

		int nyz; //Equals to ny * nz, for better index querying

		/*
		* Get the surrounding cell for a given point pos, it will not check whether the point in the bounding box for
		* speed, outT = (tx, ty, tz) is the remainder, which means:
		*     pos = the-lowest-point-position-in-the-cell + outT * cellSize
		* Thus, 0.0f < tx, ty, tz < 1.0f
		*/
		Cell getCell(const Eigen::Vector3f & pos, Eigen::Vector3f *outT = nullptr) const {
			const Eigen::Vector3f indexInFloat = (pos - minP).cwiseProduct(cellSizeReciprocal);

			//Only check the boundary in debug mode
			HairEngine_DebugAssert(MathUtility::between(pos(0), minP(0), maxP(0)));
			HairEngine_DebugAssert(MathUtility::between(pos(1), minP(1), maxP(1)));
			HairEngine_DebugAssert(MathUtility::between(pos(2), minP(2), maxP(2)));

			const auto dx = static_cast<int>(indexInFloat(0));
			const auto dy = static_cast<int>(indexInFloat(1));
			const auto dz = static_cast<int>(indexInFloat(2));

			Cell cell;
			cell.v[0] = vertexes + (dx * nyz + dy * nz + dz);
			cell.v[1] = cell.v[0] + 1;
			cell.v[2] = cell.v[0] + nz;
			cell.v[3] = cell.v[2] + 1;
			cell.v[4] = cell.v[0] + nyz;
			cell.v[5] = cell.v[4] + 1;
			cell.v[6] = cell.v[4] + nz;
			cell.v[7] = cell.v[6] + 1;

			if (outT)
				(*outT) << indexInFloat(0) - dx, indexInFloat(1) - dy, indexInFloat(2) - dz;

			return cell;
		}
	};
};

