//
// Created by vivi on 15/02/2018.
//

#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <utility>
#include <istream>


#include "../precompiled/precompiled.h"

namespace HairEngine {
	namespace MathUtility {

		/**
		 * Return the value whose has the smaller abs value
		 * @tparam T The type which supports abs(T)
		 * @param val1 The first value
		 * @param val2 The second value
		 * @return The value who has smaller value
		 */
		template <typename T>
		inline T absMin(const T & val1, const T & val2) {
			return std::abs(val1) < std::abs(val2) ? val1 : val2;
		}

		template <typename T>
		inline bool between(const T & value, const T & lowerBound, const T & upperBound) {
			return (lowerBound < value) && (value < upperBound);
		}

		template <typename T>
		inline bool betweenOrEqualToBound(const T & value, const T & lowerBound, const T & upperBound) {
			return (lowerBound <= value) && (value <= upperBound);
		}

		template <typename T>
		inline T min3(const T & a, const T & b, const T & c) {
			return std::min(std::min(a, b), c);
		}

		template <typename T>
		inline T max3(const T & a, const T & b, const T & c) {
			return std::max(std::max(a, b), c);
		}

		template <typename T>
		inline T lerp(const T & a, const T & b, float t) {
			return a + (b - a) * t;
		}

		/*
		* Get the projection vector of v projected on projectionVector, the projectionVector must be normalized.
		*/
		inline Eigen::Vector3f project(const Eigen::Vector3f & v, const Eigen::Vector3f & projectionVector) {
			Eigen::Vector3f ret;
			float signedProjectionLength = v.dot(projectionVector);
			ret = signedProjectionLength * projectionVector;
			return ret;
		}

		/*
		* Check the two vectors are colinear or not. Default tolerance is 3 degree.
		*/
		inline bool isColinear(const Eigen::Vector3f & v1, const Eigen::Vector3f & v2, float tolerance = 0.0523599f) {
			float arg = std::abs(v1.dot(v2) / (v1.norm() * v2.norm()));
			return arg > std::cos(tolerance);
		}

		/*
		* Check whether the three points are colinear with each others
		*/
		inline bool isColinear(const Eigen::Vector3f & p1, const Eigen::Vector3f & p2,
			const Eigen::Vector3f & p3, float tolerance = 0.0523599f) {
			return isColinear(p2 - p1, p3 - p2, tolerance);
		}

		/*
		* By entering a vector, returns a orthogonal vecotor. This method works for any non-zero vector.
		*/
		inline Eigen::Vector3f makeOrthogonalVector(const Eigen::Vector3f & fromVector, bool shouldNormalized) {

			Eigen::Vector3f ret;
			if (std::abs(fromVector(0)) > std::abs(fromVector(1)))
				ret = { -fromVector(2), 0, fromVector(0) };
			else
				ret = { 0, -fromVector(2), fromVector(1) };

			if (shouldNormalized)
				ret.normalize();

			return ret;
		}

		/*
		* The distance between two line segments (p0 <---> p`), (q0 <---> q1). The attached point is
		* p0 (1 - s) + s p1 and q0 (1 - s) + s q1.
		*/
		inline float lineSegmentSquaredDistance(const Eigen::Vector3f & p0, const Eigen::Vector3f & p1,
			const Eigen::Vector3f & q0, const Eigen::Vector3f & q1,
			float & s, float & t) {

			/*
			* Code from https://www.geometrictools.com/GTEngine/Include/Mathematics/GteDistSegmentSegmentExact.h
			*/
			Eigen::Vector3f p0p1 = p1 - p0;
			Eigen::Vector3f q0q1 = q1 - q0;
			Eigen::Vector3f q0p0 = p0 - q0;

			float a = p0p1.dot(p0p1);
			float b = p0p1.dot(q0q1);
			float c = q0q1.dot(q0q1);
			float d = p0p1.dot(q0p0);
			float e = q0q1.dot(q0p0);
			float f = q0p0.dot(q0p0);

			float det = a * c - b * b;
			float nd, bmd, bte, ctd, bpe, ate, btd;

			constexpr float zero = 0.0f;
			constexpr float one = 1.0f;

			if (det > zero)
			{
				bte = b * e;
				ctd = c * d;
				if (bte <= ctd)  // s <= 0
				{
					s = zero;
					if (e <= zero)  // t <= 0
					{
						// region 6
						t = zero;
						nd = -d;
						if (nd >= a)
						{
							s = one;
						}
						else if (nd > zero)
						{
							s = nd / a;
						}
						// else: s is already zero
					}
					else if (e < c)  // 0 < t < 1
					{
						// region 5
						t = e / c;
					}
					else  // t >= 1
					{
						// region 4
						t = one;
						bmd = b - d;
						if (bmd >= a)
						{
							s = one;
						}
						else if (bmd > zero)
						{
							s = bmd / a;
						}
						// else:  s is already zero
					}
				}
				else  // s > 0
				{
					s = bte - ctd;
					if (s >= det)  // s >= 1
					{
						// s = 1
						s = one;
						bpe = b + e;
						if (bpe <= zero)  // t <= 0
						{
							// region 8
							t = zero;
							nd = -d;
							if (nd <= zero)
							{
								s = zero;
							}
							else if (nd < a)
							{
								s = nd / a;
							}
							// else: s is already one
						}
						else if (bpe < c)  // 0 < t < 1
						{
							// region 1
							t = bpe / c;
						}
						else  // t >= 1
						{
							// region 2
							t = one;
							bmd = b - d;
							if (bmd <= zero)
							{
								s = zero;
							}
							else if (bmd < a)
							{
								s = bmd / a;
							}
							// else:  s is already one
						}
					}
					else  // 0 < s < 1
					{
						ate = a * e;
						btd = b * d;
						if (ate <= btd)  // t <= 0
						{
							// region 7
							t = zero;
							nd = -d;
							if (nd <= zero)
							{
								s = zero;
							}
							else if (nd >= a)
							{
								s = one;
							}
							else
							{
								s = nd / a;
							}
						}
						else  // t > 0
						{
							t = ate - btd;
							if (t >= det)  // t >= 1
							{
								// region 3
								t = one;
								bmd = b - d;
								if (bmd <= zero)
								{
									s = zero;
								}
								else if (bmd >= a)
								{
									s = one;
								}
								else
								{
									s = bmd / a;
								}
							}
							else  // 0 < t < 1
							{
								// region 0
								s /= det;
								t /= det;
							}
						}
					}
				}
			}
			else
			{
				// The segments are parallel.  The quadratic factors to R(s,t) =
				// a*(s-(b/a)*t)^2 + 2*d*(s - (b/a)*t) + f, where a*c = b^2,
				// e = b*d/a, f = |P0-Q0|^2, and b is not zero.  R is constant along
				// lines of the form s-(b/a)*t = k, and the minimum of R occurs on the
				// line a*s - b*t + d = 0.  This line must intersect both the s-axis
				// and the t-axis because 'a' and 'b' are not zero.  Because of
				// parallelism, the line is also represented by -b*s + c*t - e = 0.
				//
				// The code determines an edge of the domain [0,1]^2 that intersects
				// the minimum line, or if none of the edges intersect, it determines
				// the closest corner to the minimum line.  The conditionals are
				// designed to test first for intersection with the t-axis (s = 0)
				// using -b*s + c*t - e = 0 and then with the s-axis (t = 0) using
				// a*s - b*t + d = 0.

				// When s = 0, solve c*t - e = 0 (t = e/c).
				if (e <= zero)  // t <= 0
				{
					// Now solve a*s - b*t + d = 0 for t = 0 (s = -d/a).
					t = zero;
					nd = -d;
					if (nd <= zero)  // s <= 0
					{
						// region 6
						s = zero;
					}
					else if (nd >= a)  // s >= 1
					{
						// region 8
						s = one;
					}
					else  // 0 < s < 1
					{
						// region 7
						s = nd / a;
					}
				}
				else if (e >= c)  // t >= 1
				{
					// Now solve a*s - b*t + d = 0 for t = 1 (s = (b-d)/a).
					t = one;
					bmd = b - d;
					if (bmd <= zero)  // s <= 0
					{
						// region 4
						s = zero;
					}
					else if (bmd >= a)  // s >= 1
					{
						// region 2
						s = one;
					}
					else  // 0 < s < 1
					{
						// region 3
						s = bmd / a;
					}
				}
				else  // 0 < t < 1
				{
					// The point (0,e/c) is on the line and domain, so we have one
					// point at which R is a minimum.
					s = zero;
					t = e / c;
				}
			}

			//Floating error might yield a negative value
			return  std::max(0.0f, a * s * s - 2 * b * s * t + c * t * t + 2 * d * s - 2 * e * t + f);
		}

		/**
		* Get the CPA (closest point approach) which yields the minimum distance for the lines defined as (p0, p1) and (q0, q1).
		* That is, suppose the minimum distance interpolation point are p in line (p0, p1) and q in line (q0, q1). Then
		* The interpolation weight <s, t> should be equal to p = p0 + (p1 - q0) s, q = q0 + (q1 - q0) t.
		*
		* @param p0 The first point in Line1
		* @param p1 The second point in Line2
		* @param q0 The first point in Line1
		* @param q1 The second point in Line2
		*
		* @return A pair of float <s, t>. Indicating the interpolation weight for Line1 and Line2
		*/
		inline std::pair<float, float> linetoLineDistanceClosestPointApproach(const Eigen::Vector3f & p0, const Eigen::Vector3f & p1,
			const Eigen::Vector3f & q0, const Eigen::Vector3f & q1) {

			std::pair<float, float> r;
			Eigen::Vector3f p0p1 = p1 - p0;
			Eigen::Vector3f q0q1 = q1 - q0;
			Eigen::Vector3f q0p0 = p0 - q0;

			float a = p0p1.dot(p0p1);
			float b = p0p1.dot(q0q1);
			float c = q0q1.dot(q0q1);
			float d = p0p1.dot(q0p0);
			float e = q0q1.dot(q0p0);

			float f1 = b * b - a * c;
			r.first = (c * d - b * e) / f1;
			r.second = (b * d - a * e) / f1;

			return r;
		}

		/**
		* The line squared distance in 3D defined. Line 1 is defined by (p0, p1) and Line 2 is defined
		* by (q0, q1). Attached point for the distance vector can be computed by p0 (1 - s) + s p1 and
		* q0 ( 1 - t) + s q1
		*
		* @param p0 The first point in the Line 1
		* @param p1 The second point in the Line 1
		* @param q0 The first point in the Line 2
		* @param q1 The second point in the Line 2
		* @param outS The interpolation weight in Line1 where the interpolation point is computed as p0 (1 - s) + s p1
		* @param outT The interpolation weight in Line2 where the interpolation point is computed as q0 (1 - s) + s q1
		*
		* @return The line squared distance for two lines
		*/
		inline float linetoLineSquaredDistance(const Eigen::Vector3f & p0, const Eigen::Vector3f & p1,
			const Eigen::Vector3f & q0, const Eigen::Vector3f & q1,
			float * outS = nullptr, float *outT = nullptr) {

			std::pair<float, float> r = linetoLineDistanceClosestPointApproach(p0, p1, q0, q1);

			if (outS)
				*outS = r.first;
			if (outT)
				*outT = r.second;

			return ((p0 + r.first * (p1 - p0)) - (q0 + r.second * (q1 - q0))).squaredNorm();
		}

		/**
		* Get the closest point approach from the point to a plane. p denotes the point, and o1, o2 and o3 denotes the surface plane
		* defined with 3 poins.
		*
		* @param p The input point
		* @param o1 The first point in the plane
		* @param o2 The second point in the plane
		* @param o3 The third point in the plane
		*
		* @return A pair of float <s,t>. Where the closest point in the plane is defined as o1 + s (o2 - o1) + t (o3 - o2) or
		* (1 - s - t) o1 + s o2 + t o3.
		*/
		inline std::pair<float, float> pointToPlaneClosestPointApproach(const Eigen::Vector3f & p, const Eigen::Vector3f & o1,
			const Eigen::Vector3f & o2, const Eigen::Vector3f & o3) {
			Eigen::Vector3f op = o1 - p, d = o2 - o1, e = o3 - o1;

			float d2 = d.squaredNorm(), e2 = e.squaredNorm(), de = d.dot(e);
			float dop = d.dot(op), eop = e.dot(op);

			float f1 = 1.0f / (de * de - d2 * e2);

			float s = (e2 * dop - de * eop) * f1;
			float t = (d2 * eop - de * dop) * f1;

			return { s, t };
		}

		/**
		* Get the squared distance from a point to a plane. p denotes a point position in 3 dimensional space, and (o1, o2, o3) defines the infinity plane.
		* The function supports to output the CPA (closest point approach) where the point is define as (1 - s - t) o1 + s o2 + t o3, use outS and outT to
		* get the value.
		*
		* @param The input point position
		* @param o1 The first point in the plane
		* @param o2 The second point in the plane
		* @param o3 The third point in the plane
		*
		* @param outS The output s
		*
		* @return The squared distance from the point to the plane.
		*/
		inline float pointToPlaneSquaredDistance(const Eigen::Vector3f & p, const Eigen::Vector3f & o1, const Eigen::Vector3f & o2, const Eigen::Vector3f & o3,
			float *outS = nullptr, float *outT = nullptr) {

			std::pair<float, float> st = pointToPlaneClosestPointApproach(p, o1, o2, o3);

			if (outS)
				*outS = st.first;
			if (outT)
				*outT = st.second;

			return (p - o1 - st.first * (o2 - o1) - st.second * (o3 - o1)).squaredNorm();
		}

		/*
		* Get a affine3f without translation part
		*/
		inline Eigen::Affine3f getAffine3fWithoutTranslationPart(const Eigen::Affine3f & affine) {
			Eigen::Matrix4f affineInMatrix = affine.matrix();
			affineInMatrix(0, 3) = 0.0f;
			affineInMatrix(1, 3) = 0.0f;
			affineInMatrix(2, 3) = 0.0f;
			return Eigen::Affine3f(std::move(affineInMatrix));
		}

		/*
		* Get the midpoint of two vector
		*/
		inline Eigen::Vector3f midPoint(const Eigen::Vector3f & p1, const Eigen::Vector3f & p2) {
			return 0.5f * (p1 + p2);
		}

		/*
		* Get the triangle center of the triangle define by three points
		*/
		inline Eigen::Vector3f triangleCenter(const Eigen::Vector3f & p1, const Eigen::Vector3f & p2, const Eigen::Vector3f & p3) {
			return 1.0f / 3.0f * (p1 + p2 + p3);
		}

		inline void lerpDecompose(const Eigen::Affine3f &aff, Eigen::Vector3f &pos, Eigen::Quaternionf &rot, Eigen::Vector3f &scale)
		{
			Eigen::Matrix3f rot_mat, scale_mat;
			aff.computeRotationScaling(&rot_mat, &scale_mat);

			pos = aff.translation();
			rot = Eigen::Quaternionf(rot_mat);
			scale = scale_mat.diagonal();
		}

		inline Eigen::Affine3f lerpCompose(float alpha,
			const Eigen::Vector3f &pos0, const Eigen::Quaternionf &rot0, const Eigen::Vector3f &scale0,
			const Eigen::Vector3f &pos1, const Eigen::Quaternionf &rot1, const Eigen::Vector3f &scale1)
		{
			float one_minus_alpha = 1.0f - alpha;

			Eigen::Affine3f result;
			result.fromPositionOrientationScale(
				one_minus_alpha * pos0 + alpha * pos1,
				rot0.slerp(alpha, rot1),
				one_minus_alpha*scale0 + alpha * scale1);

			return result;
		}

		/*
		* Lerp between to Affine3f to get the correct interpolation, All affine should not contain the shear
		* components.
		*/
		inline Eigen::Affine3f lerp(float alpha, const Eigen::Affine3f &aff0, const Eigen::Affine3f &aff1)
		{
			Eigen::Vector3f pos0; Eigen::Quaternionf rot0; Eigen::Vector3f scale0;
			Eigen::Vector3f pos1; Eigen::Quaternionf rot1; Eigen::Vector3f scale1;
			lerpDecompose(aff0, pos0, rot0, scale0);
			lerpDecompose(aff1, pos1, rot1, scale1);

			if (rot0.dot(rot1) < 0.0f)
				rot1 = Eigen::Quaternionf(-rot1.w(), -rot1.x(), -rot1.y(), -rot1.z());

			return lerpCompose(alpha, pos0, rot0, scale0, pos1, rot1, scale1);
		}

		/*
		* Projection a vector to a normalized direction, get the tangent and normal vector values.
		* The direction should be normalized !!!
		*/
		inline void projection(const Eigen::Vector3f & v, const Eigen::Vector3f & dir, Eigen::Vector3f & outVn, Eigen::Vector3f & outVt) {
			outVn = v.dot(dir) * dir;
			outVt = v - outVn;
		}

		/**
		* Mass spring force computation.
		*
		* @param pos1 The first position of mass spring attached point
		* @param pos2 The second position of mass spring attached point
		* @param k The stiffness
		* @param l0 The rest length
		* @param directionVec The directional vector, nullptr indicates we use the normal spring computation
		* @param outD The optional directional matrix buffer, it is assigned as non-nullptr, a directional matrix will be written to it
		*
		* @return The sprign force
		*/
		inline Eigen::Vector3f massSpringForce(const Eigen::Vector3f & pos1, const Eigen::Vector3f & pos2,
			float k, float l0, const Eigen::Vector3f *directionVec = nullptr, Eigen::Matrix3f *outD = nullptr) {

			Eigen::Vector3f d = pos2 - pos1;
			float l = d.norm();

			// If pos2 == pos1, return 0 since we cannot determine the direction
			if (l == 0) {
				if (outD)
					*outD = Eigen::Matrix3f::Identity();
				return Eigen::Vector3f::Zero();
			}

			d.normalize();

			// Compute the directional indicator
			float s = 1.0f;
			if (directionVec != nullptr && d.dot(*directionVec) < 0.0f)
				s = -1.0f;

			// Compute the direction matrix
			if (outD)
				*outD = d * d.transpose();

			return (k * (l - s * l0)) * d;
		}

		/**
		* Eigen vectorization works for aligned Vector4f and Matrix4f, so if enabling Eigen vectorization,
		* it is faster to compute in Vector4f and Matrix4f. So massSpringForce4f is used as a alternative version
		* mass spring computation function used in Eigen Vectorization.
		*
		* @param pos1 The first position of mass spring attached point
		* @param pos2 The second position of mass spring attached point
		* @param k The stiffness
		* @param l0 The rest length
		* @param directionVec The directional vector, nullptr indicates we use the normal spring computation
		* @param outD The optional directional matrix buffer, it is assigned as non-nullptr, a directional matrix will be written to it in Vector4f
		*
		* @return The sprign force in Vector4f
		*/
		inline Eigen::Vector4f massSpringForce4f(const Eigen::Vector3f & pos1, const Eigen::Vector3f & pos2,
			float k, float l0, const Eigen::Vector3f *directionVec = nullptr, Eigen::Matrix4f *outD = nullptr) {

			Eigen::Vector4f d;
			d.segment<3>(0) = pos2 - pos1;
			d(3) = 0.0f;

			float l = d.norm();

			// If pos2 == pos1, return 0 since we cannot determine the direction
			if (l == 0) {
				if (outD)
					*outD = Eigen::Matrix4f::Identity();
				return Eigen::Vector4f::Zero();
			}

			d.normalize();

			// Compute the directional indicator
			float s = 1.0f;
			if (directionVec != nullptr) {
				Eigen::Vector4f directionVec4f;
				directionVec4f.segment<3>(0) = *directionVec;
				directionVec4f(3) = 0.0f;
				if (d.dot(directionVec4f) < 0.0f)
					s = -1.0f;
			}

			// Compute the direction matrix
			if (outD)
				*outD = d * d.transpose();

			return (k * (s * l - l0)) * d;
		}

		/**
		 * Get the bounding box from an group of points
		 * @tparam PointIterator The Iterator which dereference yields an instance of Eigen::Vector3f
		 * @param pointBegin The begin of the iterator
		 * @param pointEnd The end of the iterator
		 * @return A bounding box containning those points
		 */
		template <class PointIterator>
		inline Eigen::AlignedBox3f boundingBox(const PointIterator & pointBegin, const PointIterator & pointEnd) {

			auto it(pointBegin);
			Eigen::AlignedBox3f bbox(*(it++));

			for (; it != pointEnd; ++it) {
				bbox.extend(*it);
			}

			return bbox;
		}

		/**
		 * Scale the bounding box
		 * @param bbox The input bounding box
		 * @param scale The scale value
		 * @return The output bounding box will have the same center as the input but the diagnoal size will be scaled
		 */
		inline Eigen::AlignedBox3f scaleBoundingBox(const Eigen::AlignedBox3f & bbox, float scale) {
			Eigen::Vector3f d = (scale * 0.5f) * bbox.diagonal();

			return Eigen::AlignedBox3f(bbox.center() - d, bbox.center() + d);
		}

		/**
		 * Get the squared distance from a point the line segment
		 * @param p The point position
		 * @param x0 The first line segment vertex position
		 * @param x1 The second line segment vertex position
		 * @return The squared distance
		 */
		inline float pointToLineSegmentSquaredDistance(const Eigen::Vector3f & p,
				const Eigen::Vector3f & x0, const Eigen::Vector3f & x1) {
			Eigen::Vector3f x10 = x1 - x0;

			float t = (x1 - p).dot(x10) / x10.squaredNorm();
			t = std::max(0.0f, std::min(t, 1.0f));

			return (p - (t * x0 + (1.0f - t)*x1)).squaredNorm();
		}

		/**
		 * Get the signed distance for a point to the triangle, the out surface normal
		 * is defined by (p2 - p0) x (p1 - p0).
		 * @param p The point
		 * @param x0 The first vertex position of the triangle
		 * @param x1 The second vertex position of the triangle
		 * @param x2 The third vertex position of the triangle
		 * @param outUV The uv coordinate of the projection point on the triangle plane, in which means the projection
		 * point p' can be expressed as p' = u * x0 + v * x1 + (1 - u - v) * x2
		 * @return The signed distance from the point to triangle
		 */
		inline float pointToTriangleSignedDistance(const Eigen::Vector3f & p,
				const Eigen::Vector3f & x0, const Eigen::Vector3f & x1, const Eigen::Vector3f & x2, Eigen::Vector2f *outUV = nullptr) {
			float d = 0;
			Eigen::Vector3f x02 = x0 - x2;
			float l0 = x02.norm() + 1e-30f;
			x02 = x02 / l0;
			Eigen::Vector3f x12 = x1 - x2;
			float l1 = x12.dot(x02);
			x12 = x12 - l1 * x02;
			float l2 = x12.norm() + 1e-30f;
			x12 = x12 / l2;
			Eigen::Vector3f px2 = p - x2;

			float b = x12.dot(px2) / l2;
			float a = (x02.dot(px2) - l1*b) / l0;
			float c = 1 - a - b;

			if (outUV)
				(*outUV) << a, b;

			// normal vector of triangle. Don't need to normalize this yet.
			Eigen::Vector3f nTri = (x2 - x0).cross(x1 - x0);

			float tol = 1e-8f;

			if (a >= -tol && b >= -tol && c >= -tol) {
				// Inside the triangle
				d = (p - (a * x0 + b * x1 + c * x2)).norm();
			}
			else
			{
				d = std::min(pointToLineSegmentSquaredDistance(p, x0, x1),
						pointToLineSegmentSquaredDistance(p, x1, x2));
				d = std::min(d, pointToLineSegmentSquaredDistance(p, x0, x2));
			}

			d = sqrt(d);
			d = ((p - x0).dot(nTri) < 0.f) ? -d : d;

			return d;
		}
	}
}
