//
// Created by vivi on 07/02/2018.
//

/*
 * Some helper functions for better dealing with Eigen
 */

#pragma once

#include <string>
#include <Eigen/Eigen>
#include <sstream>
#include "VPly/vply.h"

namespace HairEngine {
	namespace EigenUtility {
		/*
		 * Convert a Vector3f into string
		 */
		inline std::string toString(const Eigen::Vector3f & v) {
			std::ostringstream s;
			s << std::showpoint << '{' << v(0) << ',' <<  v(1) << ',' << v(2) << '}';
			return s.str();
		}

		inline std::string toString(const Eigen::Matrix4f & m) {
			std::ostringstream s;
			s << std::showpoint << '{'
			  << '{' << m(0, 0) << ',' << m(0, 1) << ',' << m(0, 2) << ',' <<  m(0, 3) << '}' << ','
			  << '{' << m(1, 0) << ',' << m(1, 1) << ',' << m(1, 2) << ',' << m(1, 3) << '}' << ','
			  << '{' << m(2, 0) << ',' << m(2, 1) << ',' << m(2, 2) << ',' << m(2, 3) << '}' << ','
			  << '{' << m(3, 0) << ',' << m(3, 1) << ',' << m(3, 2) << ',' << m(3, 3) << '}'
			  << '}';
			return s.str();
		}

		inline std::string toString(const Eigen::Matrix3f & m) {
			std::ostringstream s;
			s << std::showpoint << '{'
			  << '{' << m(0, 0) << ',' << m(0, 1) << ',' << m(0, 2) << ','  << '}' << ','
			  << '{' << m(1, 0) << ',' << m(1, 1) << ',' << m(1, 2) << ','  << '}' << ','
			  << '{' << m(2, 0) << ',' << m(2, 1) << ',' << m(2, 2) << ','  << '}' << ','
			  << '}';
			return s.str();
		}

		inline std::ostream &operator<<(std::ostream & os, const Eigen::Vector3f & v) {
			os << '{' << v.x() << ',' << v.y() << ',' << v.z() << '}';
			return os;
		}

		inline std::ostream &operator<<(std::ostream & os, const Eigen::Matrix4f & m) {
			os << '{'
			  << '{' << m(0, 0) << ',' << m(0, 1) << ',' << m(0, 2) << ',' <<  m(0, 3) << '}' << ','
			  << '{' << m(1, 0) << ',' << m(1, 1) << ',' << m(1, 2) << ',' << m(1, 3) << '}' << ','
			  << '{' << m(2, 0) << ',' << m(2, 1) << ',' << m(2, 2) << ',' << m(2, 3) << '}' << ','
			  << '{' << m(3, 0) << ',' << m(3, 1) << ',' << m(3, 2) << ',' << m(3, 3) << '}'
			  << '}';
			return os;
		}

		inline std::ostream &operator<<(std::ostream & os, const Eigen::Matrix3f & m) {
			os << std::showpoint << '{'
			  << '{' << m(0, 0) << ',' << m(0, 1) << ',' << m(0, 2) << ','  << '}' << ','
			  << '{' << m(1, 0) << ',' << m(1, 1) << ',' << m(1, 2) << ','  << '}' << ','
			  << '{' << m(2, 0) << ',' << m(2, 1) << ',' << m(2, 2) << ','  << '}' << ','
			  << '}';
			return os;
		}

		/**
		 * Convert an vector 3 vector to vector 4 by adding 1 to the last dimension
		 * @param vec3 The input vector
		 * @return A vector with size 4 which equals to (vec3, 1.0f)
		 */
		inline Eigen::Vector4f fromVector3fToVector4f(const Eigen::Vector3f & vec3) {
			Eigen::Vector4f vec4;
			vec4 << vec3.x(), vec3.y(), vec3.z(), 1.0f;

			return vec4;
		}

		/**
		 * Convert an vector 4 vector to vector 3 by trimming the last dimension
		 * @param vec4 The input vector 4 vector
		 * @return A vector 3 vector
		 */
		inline Eigen::Vector3f fromVector4fToVector3f(const Eigen::Vector4f & vec4) {
			return Eigen::Vector3f(vec4(0), vec4(1), vec4(2));
		}

		/*
		 * Convert a group of Vector3f into string
		 */
		template <typename Vector3fArrayIterator>
		inline std::string toString(const Vector3fArrayIterator &arrayBegin, const Vector3fArrayIterator & arrayEnd) {
			std::ostringstream s;

			s.setf(std::ios::showpoint);

			for (Vector3fArrayIterator it = arrayBegin; it != arrayEnd; ++it) {
				const Eigen::Vector3f & v = *it;
				if (it != arrayBegin)
					s << ", ";
				s << '{' << v(0) << ',' <<  v(1) << ',' << v(2) << '}';
			}

			return s.str();
		}

		inline VPly::VPlyVector3f toVPlyVector3f(const Eigen::Vector3f & v) {
			return { v(0), v(1), v(2) };
		}

		inline VPly::VPlyMatrix4f toVPlyMatrix4f(const Eigen::Matrix4f & m) {
			return {
					m(0, 0), m(0, 1), m(0, 2), m(0, 3),
					m(1, 0), m(1, 1), m(1, 2), m(1, 3),
					m(2, 0), m(2, 1), m(2, 2), m(2, 3),
					m(3, 0), m(3, 1), m(3, 2), m(3, 3)
			};
		}

		inline VPly::VPlyVector3i toVPlyVector3i(const Eigen::Vector3i & v) {
			return { v.x(), v.y(), v.z() };
		}
	}

}