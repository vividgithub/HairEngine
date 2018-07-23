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
	}
}