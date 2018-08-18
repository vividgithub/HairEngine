//
// Created by vivi on 01/05/2018.
//

#pragma once
#include <list>
#include <forward_list>
#include <Eigen/Eigen>
#include "../precompiled/precompiled.h"

namespace HairEngine {
	/*
	* A grid structure that use for accelerate distance searching. It will subdivide a 3D space into small cells. We support
	* to insert a element T into a cell, and check elements in a specific cell.
	*/
	template <typename T>
	class FiniteGrid {

	HairEngine_Public:

		/*
		* Cell structure in FiniteGrid
		*/
		template <typename Element>
		struct Cell {
			int checkToken = 0;
			std::forward_list<Element> elements; //All the elements in that cell
		};

		using GridIndex = int;
		using GridIndex3 = Eigen::Vector3i;

		FiniteGrid(Eigen::Vector3f expectedCellSize_, Eigen::Vector3i maxCellCount_) :
			expectedCellSize(expectedCellSize_),
			maxCellCount(maxCellCount_),
			cells(new Cell<T>[maxCellCount_.x() * maxCellCount_.y() * maxCellCount_.z()]) {}

		~FiniteGrid() {
			delete[] cells;
		}

		/*
		* This method is called when the whole geometry will change, it will clear(fake) all the contents in the cell
		* and re-organize the grid to match the bound (which should be the bounding of all elements).
		*/
		void refresh(const Eigen::AlignedBox3f & bound_) {
			++token;

			constexpr const float extendFactor = 5e-3f;
			bound = bound_;
			auto diag = bound.diagonal();
			bound.min() -= extendFactor * diag;
			bound.max() += extendFactor * diag;

			Eigen::Vector3f cellCountf = bound.diagonal().cwiseQuotient(expectedCellSize);
			cellCount = {
				std::ceil(cellCountf.x()),
				std::ceil(cellCountf.y()),
				std::ceil(cellCountf.z())
			};

			for (size_t i = 0; i < 3; ++i) {
				if (cellCount(i) <= maxCellCount(i)) {
					cellSize(i) = expectedCellSize(i);
					bound.max()(i) = bound.min()(i) + cellCount(i) * cellSize(i); //Update bound.max() to match the exact cellSize and cellCount
				}
				else {
					cellCount(i) = maxCellCount(i);
					cellSize(i) = (bound.max()(i) - bound.min()(i)) / cellCount(i);
				}
			}

			cellSizeInv = cellSize.cwiseInverse();

			offsetZ = 1;
			offsetY = cellCount.z();
			offsetX = cellCount.y() * offsetY;
			offsetGlobal = cellCount.x() * offsetX;
		}

		GridIndex getIndex(const Eigen::Vector3f & pos) const {
			HairEngine_DebugAssert(bound.contains(pos));

			Eigen::Vector3i index3 = Eigen::Vector3f((pos - bound.min()).cwiseProduct(cellSizeInv)).cast<int>();
			return getIndex(index3);
		}

		GridIndex getIndex(const GridIndex3 & index3) const {
			return index3ToIndex(index3);
		}

		Eigen::AlignedBox3f getBound(const GridIndex & index) const {
			GridIndex3 index3 = indexToIndex3(index);

			Eigen::AlignedBox3f ret;
			ret.min() = index3.cast<float>().cwiseProduct(cellSize) + bound.min();
			ret.max() = ret.min() + cellSize;

			return ret;
		}

		void insert(const T & element, const GridIndex & index) {
			HairEngine_DebugAssert(token > 1); //We have refresh he grid since initialization
			HairEngine_DebugAssert(isValidIndex(index)); //Index is valid

			Cell<T> *cellPtr = cells + index;
			/*
			* Clear the dirty cell
			*/
			if (cellPtr->checkToken != token) {
				cellPtr->elements.clear();
				cellPtr->checkToken = token;
			}

			cellPtr->elements.push_front(element);
		}

		const std::forward_list<T> & getElementsInCell(const GridIndex & index) const {
			HairEngine_DebugAssert(token > 1);

			Cell<T> *cell = cells + index;
			return (isValidIndex(index) && cell->checkToken == token) ? cell->elements : empty;
		}

		bool isValidIndex(const GridIndex & index) const {
			return index >= 0 && index < offsetGlobal;
		}

		int getOffsetX() const { return offsetX; }
		int getOffsetY() const { return offsetY; }
		int getOffsetZ() const { return offsetZ; }
		int getOffsetGlobal() const { return offsetGlobal; }

		const Eigen::Vector3f & getCellSize() const {
			return cellSize;
		}

		const Eigen::Vector3f & getCellSizeInv() const {
			return cellSizeInv;
		}

	HairEngine_Private:

		GridIndex index3ToIndex(const GridIndex3 & index3) const {
			return index3.x() * offsetX + index3.y() * offsetY + index3.z();
		};

		GridIndex3 indexToIndex3(GridIndex index) const {
			GridIndex3 index3;

			index3.x() = index / offsetX;
			index %= offsetX;

			index3.y() = index / offsetY;
			index %= offsetY;

			index3.z() = index;

			return index3;
		}

		Eigen::Vector3f expectedCellSize; //The expected size of the cell, the actual cell size will not exact be the expectedCellSize, since the resolutions are limited
		Eigen::Vector3i maxCellCount; //Max cell count for each dimensions
		Eigen::Vector3f cellSize, cellSizeInv; //Current cell size, cellSize(i) >= expectedCellSize(i),
		Eigen::Vector3i cellCount; //Current cell count, cellCount(i) <= expectedCellCount(i), in each dimension

		std::forward_list<T> empty;

		int token = 1; //The token, if any cell's checkToken if not equal to token, it will get updated

		Eigen::AlignedBox3f bound;

		int offsetX, offsetY, offsetZ, offsetGlobal;

		Cell<T> *cells;
	};
}
