//
// Created by vivi on 07/02/2018.
//

/*
 * Some utility functions that handles strings
 */

#pragma once

#include <algorithm>
#include <string>

namespace HairEngine {
	namespace StringUtility {
		inline bool endswith(const std::string & s, const std::string & ending) {
			return s.size() >= ending.size() && std::equal(ending.rbegin(), ending.rend(), s.rbegin());
		}

		inline std::string getPathSeparator() {
#if defined(WIN32) || defined(_WIN32) || defined(__CYGWIN__)
			return "\\";
#else
			return "/";
#endif
		}

		/**
		 * Replace part of the string with another string
		 * 
		 * @param s Input string
		 * @param from The sub string that needs to be replaced
		 * @param to The string that is replaced with
		 * @return The replaced string
		 */
		inline std::string replace(std::string s, const std::string &from, const std::string &to) {

			if (from.empty())
				return s;

			size_t start_pos = 0;
			while ((start_pos = s.find(from, start_pos)) != std::string::npos) {
				s.replace(start_pos, from.length(), to);
				start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
			}

			return s;
		}
	}
}