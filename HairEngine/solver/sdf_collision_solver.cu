
#ifdef HAIRENGINE_ENABLE_CUDA

#include <cstdio>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "../util/cuda_helper_math.h"


__global__
void SDFCollisionSolver_cudaComputeVelocitiesKernal(const float3 *prePoses,
                                                    const float3 *poses, const int3 *indices, float3 *outVel, float tInv, int nprim) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nprim)
		return;

	int3 pids = indices[i];

	float3 p0 = (1.0f / 3.0f) * (prePoses[pids.x] + prePoses[pids.y] + prePoses[pids.z] );
	float3 p1 = (1.0f / 3.0f) * (poses[pids.x] + poses[pids.y] + poses[pids.z] );

	outVel[i] = (p1 - p0) * tInv;
}

void SDFCollisionSolver_cudaComputeVelocities(const float3 *prePoses,
                                              const float3 *poses, const int3 *indices, float3 *outVel,
                                              float tInv, int nprim, int nblock, int nthread) {
	SDFCollisionSolver_cudaComputeVelocitiesKernal<<<nblock, nthread>>>(prePoses, poses, indices, outVel, tInv, nprim);
	cudaDeviceSynchronize();

}

__global__
void SDFCollisionSolver_cudaComputeSDFGridKernal(const float3 *poses,
                                                 const int3 *indices, unsigned long long * outGrid,
                                                 int npoint, int nprim, int3 n, int margin, float3 origin, float3 d) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nprim)
		return;

	int nyz = n.y * n.z;
	int3 margin3 { margin, margin, margin };
	float3 dInv { 1.0f / d.x, 1.0f / d.y, 1.0f / d.z };

	float3 p[3] {poses[indices[i].x], poses[indices[i].y], poses[indices[i].z]};

	// Compute the bounding box
	float3 bboxMin = p[0];
	float3 bboxMax = p[0];

	bboxMin = fminf(bboxMin, p[1]);
	bboxMin = fminf(bboxMin, p[2]);

	bboxMax = fmaxf(bboxMax, p[1]);
	bboxMax = fmaxf(bboxMax, p[2]);

	int3 minIndex = max(make_int3((bboxMin - origin) * dInv) - margin3, make_int3(0));
	int3 maxIndex = min(make_int3((bboxMax - origin) * dInv) + margin3 + make_int3(1), n - 1);

	for (int ix = minIndex.x; ix <= maxIndex.x; ++ix)
		for (int iy = minIndex.y; iy <= maxIndex.y; ++iy)
			for (int iz = minIndex.z; iz <= maxIndex.z; ++iz) {

				float3 pos = origin + make_float3(ix, iy, iz) * d;

				float signedDist = cudaPointToTriangleSignedDistance(pos, p[0], p[1], p[2]);

				unsigned long long pack;

				// Pack the signed distance to the most significant 32 bit
				// and the primitive index (i) to the last significant 32 bit
				// Currently CUDA is not supported bit shift in 64-bit, so the direct way is to assign the the data
				// directly into the address. Make sure that it is Little-Endian
				uint32_t *packAddr = reinterpret_cast<uint32_t*>(&pack);
				packAddr[1] = floatflip(signedDist);
				packAddr[0] = static_cast<uint32_t>(i);


				int offset = ix * nyz + iy * n.z + iz;
				atomicMin(outGrid + offset, pack);
			}
}

void SDFCollisionSolver_cudaComputeSDFGrid(const float3 *poses, const int3 *indices, unsigned long long * outGrid,
                                                 int npoint, int nprim, int3 n, int margin, float3 origin, float3 d, int nblock, int nthread) {
	SDFCollisionSolver_cudaComputeSDFGridKernal<<<nblock, nthread>>>(poses, indices, outGrid, npoint, nprim, n, margin, origin, d);
	cudaDeviceSynchronize();
}

__device__ inline bool SDFCollisionSolver_querySDF(float3 pos, const unsigned long long *grid, const float3 *vels, float3 origin, float3 d, float3 dInv, int3 n, float *outDist, float3 *outGradient, float3 *outV = nullptr) {

	if ((pos.x < origin.x) || (pos.y < origin.y) || (pos.z < origin.z))
		return false;

	int3 maxcoor = { n.x - 1, n.y - 1, n.z - 1 };
	float3 bboxMax = origin + d * make_float3(maxcoor);

	if ((pos.x > bboxMax.x) || (pos.y > bboxMax.y) || (pos.z > bboxMax.z))
		return false;

	float3 index3f = (pos - origin) * dInv;
	int3 index3 = min( make_int3(index3f), maxcoor);

	int nyz = n.y * n.z;

	int offsets[8];
	offsets[0] = index3.x * nyz + index3.y * (n.z) + index3.z;
	offsets[1] = offsets[0] + 1;
	offsets[2] = offsets[0] + n.z;
	offsets[3] = offsets[0] + n.z + 1;
	offsets[4] = offsets[0] + nyz;
	offsets[5] = offsets[4] + 1;
	offsets[6] = offsets[4] + n.z;
	offsets[7] = offsets[4] + n.z + 1;

	float nodesSignedDist[8];
	int nodesPrimIdx[8];
	for (int i = 0; i < 8; ++i) {
		unsigned long long pack = grid[offsets[i]];
		uint32_t *packAddr = reinterpret_cast<uint32_t *>(&pack);

		nodesSignedDist[i] = ifloatflip(packAddr[1]); // Most significant 32-bit
		nodesPrimIdx[i] = static_cast<int>(packAddr[0]); // Least significant 32-bit
	}

	bool isAnyCellInvalid = false;
	bool isAllCellLargerThanContour = true;

	for (int i = 0; i < 8; ++i) {
		isAnyCellInvalid |= (nodesPrimIdx[i] == 0xffffffff);
		isAllCellLargerThanContour &= nodesSignedDist[i] > 0.0f;
	}

	if (isAnyCellInvalid || isAllCellLargerThanContour)
		return false;

	float3 t = index3f - make_float3(index3);

	float signedDist = 0.0f;
	float3 gradient = { 0.0f, 0.0f, 0.0f };

	if (outV) {
		*outV = { 0.0f, 0.0f, 0.0f };
	}

	for (int i = 0; i < 8; ++i) {

		float cx = (i & 4) ? t.x : 1.0f - t.x;
		float cy = (i & 2) ? t.y : 1.0f - t.y;
		float cz = (i & 1) ? t.z : 1.0f - t.z;

		signedDist += nodesSignedDist[i] * (cx * cy * cz);

		if (outV) {
			(*outV) += vels[nodesPrimIdx[i]] * (cx * cy * cz);
		}

		gradient.x += ((i & 4) ? 1.0f : -1.0f) * cy * cz * nodesSignedDist[i];
		gradient.y += ((i & 2) ? 1.0f : -1.0f) * cx * cz * nodesSignedDist[i];
		gradient.z += ((i & 1) ? 1.0f : -1.0f) * cx * cy * nodesSignedDist[i];
	}

	*outDist = signedDist;
	*outGradient = gradient;

	return true;
}

__global__
void SDFCollisionSolver_cudaResolveCollisionKernal(float3 *parPoses, float3 *parVels, const unsigned char *parLocalIndices, const float3 * vels,
                                                   const unsigned long long *grid, int npar, float3 origin, float3 d, int3 n, float time, float fraction, bool changeHairRoot) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= npar)
		return;

	bool unchanged = false;

	// Check whether to change hair root
	if (!changeHairRoot && parLocalIndices[i] == 0)
		unchanged = true;

	float3 vp = parVels[i];
	float3 pos = parPoses[i]; // Estimated position
	float3 estimatedPos = pos + time * vp;

	float3 dInv { 1.0f / d.x, 1.0f / d.y, 1.0f / d.z };

	float signedDist;
	float3 gradient;
	float3 v;

	if (!SDFCollisionSolver_querySDF(estimatedPos, grid, vels, origin, d, dInv, n, &signedDist, &gradient, &v))
		unchanged = true;

//	if (i == 5174) {
//		printf("pos: {%f, %f, %f}, estimatedPos: {%f, %f, %f}, vel: {%f, %f, %f}, signedDist: %f, gradient: {%f, %f, %f}, v: {%f, %f, %f}, unchanged: %d\n", pos.x, pos.y, pos.z, estimatedPos.x, estimatedPos.y, estimatedPos.z, vp.x, vp.y, vp.z, signedDist, gradient.x, gradient.y, gradient.z, v.x, v.y, v.z, unchanged);
//	}

	if (signedDist > 0.0f)
		unchanged = true;

	if (unchanged) {
		parPoses[i] = estimatedPos;
		parVels[i] = vp;
		return;
	}

	// Fix the velocity
	gradient /= length(gradient);
	float3 vpt, vpn, vt, vn;
	vpn = gradient * dot(vp, gradient);
	vpt = vp - vpn;
	vn = gradient * dot(v, gradient);
	vt = v - vn;

	float3 vrelt = vpt - vt;
	vrelt *= fmaxf(0.0f, 1.0f - fraction * length(vpn - vn) / length(vrelt));

	vp = vt + vrelt + vn;
//	vp = {0.0f, 0.0f, 0.0f};
	pos += vp * time;

//	if (i == 5174) {
//		printf("After change, pos: {%f, %f, %f}, vel: {%f, %f, %f}\n", pos.x, pos.y, pos.z, vp.x, vp.y, vp.z);
//	}

	if (SDFCollisionSolver_querySDF(pos, grid, vels, origin, d, dInv, n, &signedDist, &gradient)) {
		pos -= signedDist * gradient / length(gradient);
//		if (i == 5174) {
//			printf("After push: pos: {%f, %f, %f}, with origin signed dist: %f and gradient: {%f, %f, %f}\n", pos.x, pos.y, pos.z, signedDist, gradient.x, gradient.y, gradient.z);
//		}
	}

	parPoses[i] = pos;
	parVels[i] = vp;
}

void SDFCollisionSolver_cudaResolveCollision(float3 *parPoses, float3 *parVels, const unsigned char *parLocalIndices, const float3 * vels,
                                                   const unsigned long long *grid, int npar, float3 origin, float3 d, int3 n, float time, float fraction, bool changeHairRoot, int nblock, int nthread) {
	SDFCollisionSolver_cudaResolveCollisionKernal<<<nblock, nthread>>>(parPoses, parVels, parLocalIndices, vels, grid, npar, origin, d, n, time, fraction, changeHairRoot);
	cudaDeviceSynchronize();
}

#endif