//
// Created by vivi on 15/02/2018.
//

#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include "core/precompiled/precompiled.h"

namespace HairEngine {
	namespace MathUtility {
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
			Assert(fromVector != Eigen::Vector3f::Zero());

			Eigen::Vector3f ret;
			if (std::abs(fromVector(0)) > std::abs(fromVector(1)))
				ret = {-fromVector(2), 0, fromVector(0)};
			else
				ret = {0, -fromVector(2), fromVector(1)};

			if (shouldNormalized)
				ret.normalize();

			return ret;
		}

		/*
		 * The distance between two line segments (p0 <---> p`), (q0 <---> q1). The attached point is
		 * p0 (1 - s) + p1 and q0 (1 - s) + q1.
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
			float one_minus_alpha = 1 - alpha;

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
	}
}
