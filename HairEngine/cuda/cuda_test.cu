//
// Created by vivi on 2018/9/20.
//

#include <cuda_runtime.h>
#include <random>
#include <iostream>

/**
 * A simple cuda kernal to test whether cuda is enabling and can be used
 * @param a A float array
 * @param b A float array
 * @param c A float array
 */
__global__
void simpleCUDATestKernal(const float *a, const float *b, float *c) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

void simpleCUDATest() {

	int testSize = 1 << 20;

	std::cout << "[simpleCUDATest] Begin..." << std::endl;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(1.0f, 2.0f);

	float *a = new float[testSize];
	float *b = new float[testSize];
	float *c = new float[testSize];

	for (int i = 0; i < testSize; ++i) {
		a[i] = dis(gen);
		b[i] = dis(gen);
		c[i] = dis(gen);
	}

	float *da, *db, *dc;
	cudaMalloc(&da, sizeof(float) * testSize);
	cudaMalloc(&db, sizeof(float) * testSize);
	cudaMalloc(&dc, sizeof(float) * testSize);

	cudaMemcpy(da, a, sizeof(float) * testSize, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, sizeof(float) * testSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dc, c, sizeof(float) * testSize, cudaMemcpyHostToDevice);

	int threadSize = 256;
	int threadBlockSize = (testSize + threadSize - 1) / threadSize;

	simpleCUDATestKernal<<<threadBlockSize, threadSize>>>(da, db, dc);

	delete[] a;
	delete[] b;
	delete[] c;

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	std::cout << "[simpleCUDATest] Done..." << std::endl;
}