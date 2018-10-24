//
// Created by vivi on 2018/10/21. Simple matrix and vector opereations.
//

#pragma once

#include "cuda_helper_math.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>

namespace HairEngine {

	struct Mat3 {
		float _val[9];

		__device__ __host__ __forceinline__
		float &operator()(int r, int c) {
			return _val[r * 3 + c];
		}

		__device__ __host__ __forceinline__
		float operator()(int r, int c) const {
			return _val[r * 3 + c];
		}

		template <int Row, int Col>
		__device__ __host__ __forceinline__
		float at() const {
			return _val[Row * 3 + Col];
		}

		template <int Row, int Col>
		__device__ __host__ __forceinline__
		float &at() {
			return _val[Row * 3 + Col];
		}

		__device__ __host__ __forceinline__
		float &operator[](int i) {
			return _val[i];
		}

		__device__ __host__ __forceinline__
		float operator[](int i) const {
			return _val[i];
		}

		Mat3() = default;

		__device__ __host__ __forceinline__
		constexpr Mat3(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22) :
				_val {a00, a01, a02, a10, a11, a12, a20, a21, a22} {}

		__device__ __host__ __forceinline__
		constexpr Mat3(const Mat3 & rhs):
			Mat3(rhs._val[0], rhs._val[1], rhs._val[2], rhs._val[3], rhs._val[4], rhs._val[5], rhs._val[6], rhs._val[7], rhs._val[8]) {}

		__device__ __host__ __forceinline__
		Mat3 & operator=(const Mat3 & rhs) {
			memcpy((float*)_val, (float*)rhs._val, sizeof(float) * 9);
			return *this;
		}

		__device__ __host__ __forceinline__
		Mat3 &operator-=(const Mat3 & rhs) {
			_val[0] -= rhs._val[0];
			_val[1] -= rhs._val[1];
			_val[2] -= rhs._val[2];
			_val[3] -= rhs._val[3];
			_val[4] -= rhs._val[4];
			_val[5] -= rhs._val[5];
			_val[6] -= rhs._val[6];
			_val[7] -= rhs._val[7];
			_val[8] -= rhs._val[8];

			return *this;
		}

		__device__ __host__ __forceinline__
		Mat3 &operator+=(const Mat3 & rhs) {
			_val[0] += rhs._val[0];
			_val[1] += rhs._val[1];
			_val[2] += rhs._val[2];
			_val[3] += rhs._val[3];
			_val[4] += rhs._val[4];
			_val[5] += rhs._val[5];
			_val[6] += rhs._val[6];
			_val[7] += rhs._val[7];
			_val[8] += rhs._val[8];

			return *this;
		}

		__device__ __host__ __forceinline__
		Mat3 inverse() const {
			const auto & a = _val;

			Mat3 ret {
				a[4] * a[8] - a[5] * a[7],
				a[2] * a[7] - a[1] * a[8],
				a[1] * a[5] - a[2] * a[4],
				a[5] * a[6] - a[3] * a[8],
				a[0] * a[8] - a[2] * a[6],
				a[2] * a[3] - a[0] * a[5],
				a[3] * a[7] - a[4] * a[6],
				a[1] * a[6] - a[0] * a[7],
				a[0] * a[4] - a[1] * a[3]
			};

			auto & b = ret._val;
			auto di = 1.0f / (a[0] * b[0] + a[1] * b[3] + a[2] * b[6]); // Determinant inverse

			b[0] *= di; b[1] *= di; b[2] *= di;
			b[3] *= di; b[4] *= di; b[5] *= di;
			b[6] *= di; b[7] *= di; b[8] *= di;

			return ret;
		}

		__device__ __host__ __forceinline__
		void print() const {
			printf("{{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}}", _val[0], _val[1], _val[2], _val[3], _val[4], _val[5], _val[6], _val[7], _val[8]);
		}

		__device__ __host__ __forceinline__
		constexpr static Mat3 Zero() {
			return { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		}

		__device__ __host__ __forceinline__
		Mat3 & asZero() {
			_val[0] = 0.0f; _val[1] = 0.0f; _val[2] = 0.0f;
			_val[3] = 0.0f; _val[4] = 0.0f; _val[5] = 0.0f;
			_val[6] = 0.0f; _val[7] = 0.0f; _val[8] = 0.0f;
			return *this;
		}

		__device__ __host__ __forceinline__
		constexpr static Mat3 Identity() {
			return { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
		}

		__device__ __host__ __forceinline__
		Mat3 & asIdentity() {
			_val[0] = 1.0f; _val[1] = 0.0f; _val[2] = 0.0f;
			_val[3] = 0.0f; _val[4] = 1.0f; _val[5] = 0.0f;
			_val[6] = 0.0f; _val[7] = 0.0f; _val[8] = 1.0f;
			return *this;
		}

		__device__ __host__ __forceinline__
		static Mat3 Diagonal(float val) {
			return { val, 0.0f, 0.0f, 0.0f, val, 0.0f, 0.0f, 0.0f, val };
		}

		__device__ __host__ __forceinline__
		Mat3 & asDiagonal(float val) {
			_val[0] = val; _val[1] = 0.0f; _val[2] = 0.0f;
			_val[3] = 0.0f; _val[4] = val; _val[5] = 0.0f;
			_val[6] = 0.0f; _val[7] = 0.0f; _val[8] = val;
			return *this;
		}

		__device__ __host__ __forceinline__
		static Mat3 FromDirection(float3 d) {
			float xx = d.x * d.x;
			float xy = d.x * d.y;
			float xz = d.x * d.z;
			float yy = d.y * d.y;
			float yz = d.y * d.z;
			float zz = d.z * d.z;

			return { xx, xy, xz, xy, yy, yz, xz, yz, zz };
		}

		__device__ __host__ __forceinline__
		Mat3 & asDirectionMatrix(float3 d) {
			at<0, 0>() = d.x * d.x;
			at<0, 1>() = at<1, 0>() = d.x * d.y;
			at<0, 2>() = at<2, 0>() = d.x * d.z;
			at<1, 1>() = d.y * d.y;
			at<1, 2>() = at<2, 1>() = d.y * d.z;
			at<2, 2>() = d.z * d.z;

			return *this;
		}
	};

	__device__ __host__ __forceinline__
	Mat3 operator*(const Mat3 & a, const Mat3 & b) {
		return {
			a.at<0, 0>() * b.at<0, 0>() + a.at<0, 1>() * b.at<1, 0>() + a.at<0, 2>() * b.at<2, 0>(),
			a.at<0, 0>() * b.at<0, 1>() + a.at<0, 1>() * b.at<1, 1>() + a.at<0, 2>() * b.at<2, 1>(),
			a.at<0, 0>() * b.at<0, 2>() + a.at<0, 1>() * b.at<1, 2>() + a.at<0, 2>() * b.at<2, 2>(),

			a.at<1, 0>() * b.at<0, 0>() + a.at<1, 1>() * b.at<1, 0>() + a.at<1, 2>() * b.at<2, 0>(),
			a.at<1, 0>() * b.at<0, 1>() + a.at<1, 1>() * b.at<1, 1>() + a.at<1, 2>() * b.at<2, 1>(),
			a.at<1, 0>() * b.at<0, 2>() + a.at<1, 1>() * b.at<1, 2>() + a.at<1, 2>() * b.at<2, 2>(),

			a.at<2, 0>() * b.at<0, 0>() + a.at<2, 1>() * b.at<1, 0>() + a.at<2, 2>() * b.at<2, 0>(),
			a.at<2, 0>() * b.at<0, 1>() + a.at<2, 1>() * b.at<1, 1>() + a.at<2, 2>() * b.at<2, 1>(),
			a.at<2, 0>() * b.at<0, 2>() + a.at<2, 1>() * b.at<1, 2>() + a.at<2, 2>() * b.at<2, 2>()
		};
	}

	__device__ __host__ __forceinline__
	float3 operator*(const Mat3 & a, const float3 & b) {
		return {
			a.at<0, 0>() * b.x + a.at<0, 1>() * b.y + a.at<0, 2>() * b.z,
			a.at<1, 0>() * b.x + a.at<1, 1>() * b.y + a.at<1, 2>() * b.z,
			a.at<2, 0>() * b.x + a.at<2, 1>() * b.y + a.at<2, 2>() * b.z
		};
	}
}
