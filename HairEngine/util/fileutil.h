//
// Created by vivi on 07/02/2018.
//

/*
 * Some file reading and writing helpful functions
 */

#pragma once

#include <fstream>
#include <cinttypes>
#include <Eigen/Eigen>

namespace HairEngine {
	namespace FileUtility {
		inline int32_t binaryReadInt32(std::istream & fin) {
			int32_t tmp;
			fin.read(reinterpret_cast<char*>(&tmp), sizeof(int32_t));
			return tmp;
		}

		inline float binaryReadReal32(std::istream & fin) {
			float tmp;
			fin.read(reinterpret_cast<char*>(&tmp), sizeof(float));
			return tmp;
		}

		inline Eigen::Vector3f binaryReadVector3f(std::istream & fin) {
			float storage[3];

			Eigen::Vector3f tmp;

			fin.read(reinterpret_cast<char*>(storage), sizeof(float) * 3);
			tmp << storage[0], storage[1], storage[2];

			return std::move(tmp);
		}

		inline bool binaryWriteInt32(std::ostream & fout, const int32_t & value) {
			fout.write((char*)(&value), sizeof(int32_t));
			return static_cast<bool>(fout);
		}

		inline bool binaryWriteReal32(std::ostream & fout, const float & value) {
			fout.write((char*)(&value), sizeof(float));
			return static_cast<bool>(fout);
		}

		inline bool binaryWriteVector3f(std::ostream & fout, const Eigen::Vector3f & vector3f) {
			float tmp[3] = { vector3f(0), vector3f(1), vector3f(2) };
			fout.write(reinterpret_cast<char*>(tmp), sizeof(float) * 3);
			return static_cast<bool>(fout);
		}
	}
}