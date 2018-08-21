#include <vector>
#include <algorithm>
#include <iostream>
#include <queue>
#include <utility>

#include "CompactNSearch.h"
#include "VPly/vply.h"
#include "../util/mathutil.h"
#include "../old/finite_grid.h"

#include "visualizer.h"
#include "segment_knn_solver.h"

namespace HairEngine {

	class HairContactsImpulseSolverVisualizer;

	/**
	 * The solver that resolve hair contacts by using impulse based spring forces 
	 * in semi implicit euler. It will add the force to the "im" property of the particles.
	 * The HairContactsImpulseSolver needs the SegmentKNNSolver for finding the k nearest neighbour 
	 * and add forces between them.
	 */
	class HairContactsImpulseSolver: public Solver {

	friend class HairContactsAndCollisionImpulseSolverVisualizer;

	HairEngine_Public:
		/**
		 * Constructor
		 * 
		 * @param segmentKnnSolver The SegmentKNNSolver used for neighbour search, added it before this solver to 
		 * build the knn acceleration structure.
		 * @param creatingDistance The creating distance of contact spring 
		 * @param breakingDistance The breaking distance of the contact spring
		 * @param maxContactPerSegment The limitation of max contact per segment
		 * @param kContactSpring The stiffness of the contact 
		 */
		HairContactsImpulseSolver(SegmentKNNSolver *segmentKnnSolver, float creatingDistance, float breakingDistance, int maxContactPerSegment, float kContactSpring): 
			segmentKnnSolver(segmentKnnSolver), 
			kContactSpring(kContactSpring),
			creatingDistance(creatingDistance),
			breakingDistance(breakingDistance),
			breakingDistanceSquared(breakingDistance * breakingDistance),
			maxContactPerSegment(maxContactPerSegment)
		{}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			// Setup the usedBuffers, assign a individual used buffer for each thread to avoid race condition
			for (int i = 0; i < static_cast<int>(ParallismUtility::getOpenMPMaxHardwareConcurrency()); ++i) {
				usedBuffers.emplace_back(hair.nsegment, -1);
			}

			// Setup the contact springs
			HairEngine_AllocatorAllocate(contactSprings, hair.nsegment * maxContactPerSegment);

			// Setup the ndirected and nundirected
			ncontacts = std::vector<int>(hair.nsegment, 0);
		}

		void tearDown() override {
			HairEngine_AllocatorDeallocate(contactSprings, hair->nsegment * maxContactPerSegment);
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {

			// Erase all the distance larger the r, don't parallel since we modify nundirected[_.idx2]
			ParallismUtility::parallelForWithThreadIndex(0, hair.nsegment, [this, &hair] (int idx1, int threadId) {
				const auto range = getContactSpringRange(idx1);
				auto seg1 = hair.segments + idx1;
				auto & usedBuffer = usedBuffers[threadId];

				std::fill(usedBuffer.begin(), usedBuffer.end(), -1);

				const auto removeEnd = std::remove_if(range.first, range.second, [this, seg1, &hair](const ContactSpringInfo & _) -> bool {
					return (seg1->midpoint() - (hair.segments + _.idx2)->midpoint()).squaredNorm() > breakingDistanceSquared;
				});

				// Update the nContacts now
				ncontacts[idx1] = static_cast<int>(removeEnd - range.first);

				// Compute the force of all undeleted spring and set the usedBuffer
				Eigen::Vector3f force = Eigen::Vector3f::Zero();

				for (auto _ = range.first; _ != removeEnd; ++_) {
					auto seg2 = hair.segments + _->idx2;
					force += MathUtility::massSpringForce(seg1->midpoint(), seg2->midpoint(), kContactSpring, _->l0);

					usedBuffer[_->idx2] = idx1;
				}

				syncLock.lock();
				seg1->p1->impulse += force;
				seg1->p2->impulse += force;
				syncLock.unlock();

				// Add addtional spring
				const int nneeds = std::min(segmentKnnSolver->getNNeighbourForSegment(idx1), maxContactPerSegment - ncontacts[idx1]);
				for (int i = 0; i < nneeds; ++i) {
					const int idx2 = segmentKnnSolver->getNeighbourIndexForSegment(idx1, i);
					const auto seg2 = hair.segments + idx2;

					if (usedBuffer[idx2] != idx1 && seg2->strandIndex() != seg1->strandIndex()) {
						const float l02 = (seg2->midpoint() - seg1->midpoint()).squaredNorm();
						if (l02 < creatingDistance)
							std::allocator<ContactSpringInfo>().construct(range.first + (ncontacts[idx1]++), idx2, std::sqrt(l02));
					}
				}
			});
		}

	HairEngine_Protected :

		struct ContactSpringInfo {
			int idx2; ///< The index for another endpoint
			float l0; ///< The rest length when creating 

			ContactSpringInfo(int idx2, float l0): idx2(idx2), l0(l0) {}
		};

		SegmentKNNSolver *segmentKnnSolver;
		float kContactSpring; ///< The stiffness of the contact spring

		ContactSpringInfo *contactSprings; ///< Index array of the contacts spring
		std::vector<std::vector<int>> usedBuffers; ///< Used in iteration to indicate whether the spring has been created
		std::vector<int> ncontacts; ///< How many contact spring is stored in the range (contactSprings[i * maxContactPerSegment], contactSpring[(i+1) * maxContactPerSegment] )

		float creatingDistance;
		float breakingDistance;
		float breakingDistanceSquared;
		int maxContactPerSegment;

		CompactNSearch::Spinlock syncLock; ///< Use to sync the thread

		std::pair<ContactSpringInfo *, ContactSpringInfo *> getContactSpringRange(int segmentIndex) const {
			std::pair<ContactSpringInfo *, ContactSpringInfo *> ret;
			ret.first = contactSprings + segmentIndex * maxContactPerSegment;;
			ret.second = ret.first + ncontacts[segmentIndex];

			return ret;
		}

	};

	class HairContactsImpulseSolverOld: public Solver {

	HairEngine_Public:
		HairContactsImpulseSolverOld(float creatingDistance, float breakingDistance, float gridSize, int maxGridResolution, int maxContactPerSegment, float kContact):
			creatingDistance(creatingDistance), 
			breakingDistance(breakingDistance),
			kContact(kContact),
			maxContactPerSegment(maxContactPerSegment),
			grid(Eigen::Vector3f(gridSize, gridSize, gridSize), Eigen::Vector3i(maxGridResolution, maxGridResolution, maxGridResolution))
		{}

		void setup(const Hair& hair, const Eigen::Affine3f& currentTransform) override {
			ds.resize(hair.nsegment);
			dInvs.resize(hair.nsegment);

			HairEngine_AllocatorAllocate(contactSprings, hair.nsegment * maxContactPerSegment);

			for (int i = 0; i < static_cast<int>(ParallismUtility::getOpenMPMaxHardwareConcurrency()); ++i) {
				usedBuffers.emplace_back(hair.nsegment, -1);
			}

			nundirected = std::vector<int>(hair.nsegment, 0);
			ndirected = std::vector<int>(hair.nsegment, 0);
		}

		void tearDown() override {
			HairEngine_AllocatorDeallocate(contactSprings, hair->nsegment * maxContactPerSegment);
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {
			/*
			* 	Based on the strategy we use, a line segment has a continous record index, so that we could iterate one
			* 	line segment with a index iterator. We use a heap to do this
			*/
			const static auto distanceComparator = [](const SegmentComparatorStruct & comparator1, const SegmentComparatorStruct & comparator2) -> bool {
				//We want to ensure that the segment are in order in test, to avoid equal distance for two segments
				// (always happen when for two continous segments in the same strand)
#ifndef HAIRENGINE_TEST
				return comparator1.distanceSquared < comparator2.distanceSquared;
#else
				return comparator1.distanceSquared < comparator2.distanceSquared
					|| (comparator1.distanceSquared == comparator2.distanceSquared && comparator1.l2->globalIndex < comparator2.l2->globalIndex);
#endif
			};

			float creatingDistanceSquared = creatingDistance * creatingDistance;
			float breakingDistanceSquared = breakingDistance * breakingDistance;

			/*
			* Grid insertion bounding box update
			*/
			std::cout << "HairContactsImpulseSolverOld: " << "Grid insertion..." << std::endl;
			Eigen::AlignedBox3f bound(hair.particles[0].pos);
			for (int i = 1; i < hair.nparticle; ++i)
				bound.extend(hair.particles[i].pos);
			grid.refresh(bound);

			std::vector<int> offsets[7];
			Eigen::Vector3i w;
			for (int i = 0; i < 3; ++i) {
				w(i) = static_cast<int>(std::ceil(creatingDistance * grid.getCellSizeInv()(i)));
			}

			for (int ix = 0; ix < w.x() + 1; ix = ix > 0 ? -ix : -ix + 1)
				for (int iy = 0; iy < w.y() + 1; iy = iy > 0 ? -iy : -iy + 1)
					for (int iz = 0; iz < w.z() + 1; iz = iz > 0 ? -iz : -iz + 1)
						offsets[0].push_back(grid.getIndex(Eigen::Vector3i(ix, iy, iz)));

			for (int ix = 0; ix < w.x() + 1; ix = ix > 0 ? -ix : -ix + 1) {
				for (int iy = 0; iy < w.y() + 1; iy = iy > 0 ? -iy : -iy + 1) {
					offsets[5].push_back(grid.getIndex(Eigen::Vector3i(ix, iy, w.z())));
					offsets[6].push_back(grid.getIndex(Eigen::Vector3i(ix, iy, -w.z())));
				}
			}

			for (int iy = 0; iy < w.y() + 1; iy = iy > 0 ? -iy : -iy + 1) {
				for (int iz = 0; iz < w.z() + 1; iz = iz > 0 ? -iz : -iz + 1) {
					offsets[1].push_back(grid.getIndex(Eigen::Vector3i(w.x(), iy, iz)));
					offsets[2].push_back(grid.getIndex(Eigen::Vector3i(-w.x(), iy, iz)));
				}
			}

			for (int ix = 0; ix < w.x() + 1; ix = ix > 0 ? -ix : -ix + 1) {
				for (int iz = 0; iz < w.z() + 1; iz = iz > 0 ? -iz : -iz + 1) {
					offsets[3].push_back(grid.getIndex(Eigen::Vector3i(ix, w.y(), iz)));
					offsets[4].push_back(grid.getIndex(Eigen::Vector3i(ix, -w.y(), iz)));
				}
			}

			// Compute ds and dInvs
			ParallismUtility::parallelFor(0, hair.nsegment, [this, &hair] (int i) {
				ds[i] = hair.segments[i].d();
				dInvs[i] = ds[i].cwiseInverse();
			});

			std::vector<SegmentGridRecord> records;

			/*
			* Grid insertion
			*/
			for (int i = 0; i < hair.nsegment; ++i) {

				auto seg = hair.segments + i;

				const auto & d = ds[i];
				const auto & dInv = dInvs[i];
				const int moveX = (d.x() > 0.0f) ? 1 : -1;
				const int moveY = (d.y() > 0.0f) ? 1 : -1;
				const int moveZ = (d.z() > 0.0f) ? 1 : -1;

				SegmentGrid::GridIndex index = grid.getIndex(seg->p1->pos);
				Eigen::AlignedBox3f ib = grid.getBound(index);

				grid.insert(seg, index);
				records.emplace_back(seg, index, 0);

				float tx = (d.x() > 0.0f) ? (ib.max().x() - seg->p1->pos.x()) * dInv.x()
					: (d.x() != 0.0f) ? (ib.min().x() - seg->p1->pos.x()) * dInv.x()
					: 1.0e30f;
				float ty = (d.y() > 0.0f) ? (ib.max().y() - seg->p1->pos.y()) * dInv.y()
					: (d.y() != 0.0f) ? (ib.min().y() - seg->p1->pos.y()) * dInv.y()
					: 1.0e30f;
				float tz = (d.z() > 0.0f) ? (ib.max().z() - seg->p1->pos.z()) * dInv.z()
					: (d.z() != 0.0f) ? (ib.min().z() - seg->p1->pos.z()) * dInv.z()
					: 1.0e30f;

				//A cell's width (in x, y, z) in the segment's length measurement
				float dtx = std::abs(grid.getCellSize().x() * dInv.x());
				float dty = std::abs(grid.getCellSize().y() * dInv.y());
				float dtz = std::abs(grid.getCellSize().z() * dInv.z());

				float t;
				int minTIndex;
				int indicator;
				do {
					//Update t to the min{tx, ty, tz}
					minTIndex = 0; t = tx;
					if (t > ty) {
						minTIndex = 1; t = ty;
					}
					if (t > tz) {
						minTIndex = 2; t = tz;
					}

					if (t > 1.0f)
						break;

					switch (minTIndex) {
					case 0: //Move along x axis
						index += moveX > 0 ? grid.getOffsetX() : -grid.getOffsetX();
						indicator = moveX > 0 ? 1 : 2;
						tx += dtx;

						break;
					case 1:
						index += moveY > 0 ? grid.getOffsetY() : -grid.getOffsetY();
						indicator = moveY > 0 ? 3 : 4;
						ty += dty;

						break;
					case 2:
						index += moveZ > 0 ? grid.getOffsetZ() : -grid.getOffsetZ();
						indicator = moveZ > 0 ? 5 : 6;
						tz += dtz;

						break;
					}

					//Insert it to the new grid
					grid.insert(seg, index);
					records.emplace_back(seg, index, indicator);

				} while (true);
			}

			/*
			* Solve hair contacts
			*/

			std::cout << "HairContactsImpulseSolverOld: " << "Solve hair contacts..." << std::endl;
			/*
			* For each segment pairs (currently exisits), use "compute()" to get the current length.
			* Clear all the segment with distance larger than the breaking distance
			*/
			for (int idx1 = 0; idx1 < hair.nsegment; ++idx1) {
				auto seg1 = hair.segments + idx1;
				Eigen::Vector3f midpoint1 = (hair.segments + idx1)->midpoint();

				const auto contactSpringsBeginPtr = getContactSpringsBeginPtr(idx1);
				const auto contactSpringsEndPtr = contactSpringsBeginPtr + ndirected[idx1];

				auto deletedEnd = contactSpringsBeginPtr;
				for (auto l = contactSpringsBeginPtr; l < contactSpringsEndPtr; ++l) {

					Eigen::Vector3f midpoint2 = (hair.segments + l->idx2)->midpoint();

					if ((midpoint2 - midpoint1).squaredNorm() >= breakingDistanceSquared) {

						--nundirected[idx1];
						--nundirected[l->idx2];

						std::allocator<ContactSpringInfo>().destroy(l);
					}
					else {
						if (deletedEnd != l)
							*deletedEnd = *l;
						++deletedEnd;
					}
				}

				ndirected[idx1] = static_cast<int>(deletedEnd - contactSpringsBeginPtr);
			}

			auto & segmentCheck = usedBuffers[0]; 
			std::fill(segmentCheck.begin(), segmentCheck.end(), -1);

			for (int recordBeginIndex = 0, recordEndIndex = 1; recordBeginIndex < records.size(); recordBeginIndex = recordEndIndex) {

				while (recordEndIndex < records.size() && records[recordEndIndex].seg == records[recordBeginIndex].seg)
					++recordEndIndex;

				auto seg1 = records[recordBeginIndex].seg;
				int idx1 = seg1->globalIndex;

				if (nundirected[idx1] >= maxContactPerSegment)
					continue;

				const int nneeds = maxContactPerSegment - nundirected[idx1];
				if (nneeds <= 0)
					continue;

				//Fill the contactSpringCheck for the corresponding segment
				const auto contactSpringsBeginPtr = getContactSpringsBeginPtr(idx1);
				const auto contactSpringsEndPtr = contactSpringsBeginPtr + ndirected[idx1];

				for (auto l = contactSpringsBeginPtr; l != contactSpringsEndPtr; ++l)
					segmentCheck[l->idx2] = idx1;

				std::priority_queue<SegmentComparatorStruct, std::vector<SegmentComparatorStruct>, decltype(distanceComparator)> heap(distanceComparator);

				//Iterate over recordBeginIndex to recordEndIndex, find all the "potential connected" segment pointer
				for (int i = recordBeginIndex; i < recordEndIndex; ++i) {
					const auto & record = records[i];
					for (int offset : offsets[record.indicator]) {
						const auto & segs = grid.getElementsInCell(record.index + offset); //An invalid index will yield empty element list
						for (auto seg2 : segs) {

							int idx2 = seg2->globalIndex;

							if (idx2 <= idx1 || nundirected[idx2] >= maxContactPerSegment || segmentCheck[idx2] == idx1)
								continue;

							segmentCheck[idx2] = idx1;

							SegmentComparatorStruct comparator;
							comparator.l2 = seg2;
							comparator.distanceSquared = MathUtility::lineSegmentSquaredDistance(seg1->p1->pos, seg1->p2->pos, seg2->p1->pos, seg2->p2->pos, comparator.t1, comparator.t2);

							if (comparator.distanceSquared < creatingDistanceSquared) {
								heap.push(comparator);
								if (heap.size() > nneeds)
									heap.pop();
							}
						}
					}
				}

				while (!heap.empty()) {
					const auto & comparator = heap.top();

					const int idx2 = nundirected[comparator.l2->globalIndex];

					std::allocator<ContactSpringInfo>().construct(contactSpringsBeginPtr + ndirected[idx1],
						comparator.l2->globalIndex, comparator.t1, comparator.t2, std::sqrt(comparator.distanceSquared));

					++ndirected[idx1];
					++nundirected[idx1];
					++nundirected[idx2];

					heap.pop();
				}
			}

			for (int idx1 = 0; idx1 < hair.nsegment; ++idx1) {

				auto seg1 = hair.segments + idx1;

				const auto contactSpringsBeginPtr = getContactSpringsBeginPtr(idx1);
				const auto contactSpringsEndPtr = contactSpringsBeginPtr + ndirected[idx1];

				for (auto l = contactSpringsBeginPtr; l != contactSpringsEndPtr; ++l) {

					auto idx2 = l->idx2;
					auto seg2 = hair.segments + idx2;

					Eigen::Vector3f force = MathUtility::massSpringForce(
						MathUtility::lerp(seg1->p1->pos, seg1->p2->pos, l->t1),
						MathUtility::lerp(seg2->p1->pos, seg2->p2->pos, l->t2),
						kContact, l->l0
					);

					seg1->p1->impulse += force * (1 - l->t1);
					seg1->p2->impulse += force * l->t1;
					seg2->p1->impulse -= force * (1 - l->t2);
					seg2->p2->impulse -= force * l->t2;
				}
			}
		}

	HairEngine_Protected :
		float creatingDistance;
		float breakingDistance;
		float kContact;
		float gridSize;
		float maxGridResolution;
		int maxContactPerSegment;

		struct SegmentComparatorStruct {
			float t1, t2;
			float distanceSquared;
			Hair::Segment::Ptr l2;
		};

		using SegmentGrid = FiniteGrid<Hair::Segment::Ptr>;

		struct SegmentGridRecord {
			Hair::Segment::Ptr seg; // The segment is inserted
			SegmentGrid::GridIndex index; // The grid index for the segment
			int indicator; // The indicator for insertion (0 for initialization, 1(-1) for x, 2(-2) for y, 3(-3) for z)

			SegmentGridRecord(Hair::Segment::Ptr seg, const SegmentGrid::GridIndex & index, int indicator) :
				seg(seg),
				index(index),
				indicator(indicator) {}
		};

		struct ContactSpringInfo {
			int idx2; ///< The index for another endpoint
			float t1, t2; ///< The interpolation point
			float l0; ///< The rest length when creating 

			ContactSpringInfo(int idx2, float t1, float t2, float l0) : idx2(idx2), t1(t1), t2(t2), l0(l0) {}
		};

		ContactSpringInfo *getContactSpringsBeginPtr(int segmentIndex) const {
			return contactSprings + segmentIndex * maxContactPerSegment;
		}

		ContactSpringInfo *contactSprings; ///< Index array of the contacts spring, size equals to nsegment * maxContactPerSegment
		std::vector<std::vector<int>> usedBuffers; ///< Used in iteration to indicate whether the spring has been created

		std::vector<int> nundirected; ///< The total contacts spring count for a segment
		std::vector<int> ndirected; ///< The total contacts spring stored in a segment (we store a ContactSpringInfo in the segment with lower global index)

		std::vector<Eigen::Vector3f> ds; ///< The computed segment.d() for all the line segments in hair (for reuse)
		std::vector<Eigen::Vector3f> dInvs; ///< The inverse direction of the line segments

		SegmentGrid grid;
	};
}
