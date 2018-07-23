#pragma once

#ifdef HAIRENGINE_ENABLE_VPBRT

#include <ostream>
#include <sstream>

#include "VPly/vply.h"

#include "../precompiled/precompiled.h"
#include "../util/stringutil.h"
#include "integration_info.h"
#include "solver.h"

namespace HairEngine {

	/**
	 * Visualizer is a special solver, which doesn't change the hair geometry but 
	 * output some useful information for debugging and visualization for hair into files in frames.
	 */
	class Visualizer : public Solver {

	HairEngine_Public:
		/**
		 * Constructor
		 * 
		 * @param directory Specify the directory that the informatin dumps to. For every simulation step,
		 * the visualizer will create an file with name specified by filenameTemplate.
		 * @param filenameTemplate The template specifies the real filename for a certain simulation frame. 
		 * Currently, we use "${F}" to specify current frame number. So a "${F}" substring will be replaced to 
		 * the real frame number at different simulation step to avoid name conflict
		 */
		Visualizer(const std::string & directory, const std::string & filenameTemplate)
			: directory(directory), filenameTemplate(filenameTemplate) {

			// Append path seprator if needed
			if (!StringUtility::endswith(this->directory, StringUtility::getPathSeparator()))
				this->directory += StringUtility::getPathSeparator();
		}

		void solve(Hair& hair, const IntegrationInfo& info) final override {
			auto filepath = getFilepath(info);

			std::fstream fout(filepath, std::ios::out | std::ios::binary);
			visualize(fout, hair, info);

			fout.close();
		}

		/**
		 * It is called by the solver which the dumped file stream is automatically 
		 * opened and handled in the solver. Subclass needs to implement it to write information 
		 * to the "os".
		 * 
		 * @param os The stream that writes to
		 * @param hair The hair geometry, we don't add "const" prefix to hair to give you enough freedom
		 * @param info The integration infomation
		 */
		virtual void visualize(std::ostream & os, Hair & hair, const IntegrationInfo & info) = 0;

	HairEngine_Protected:
		std::string directory;
		std::string filenameTemplate;

		std::string getFilepath(const IntegrationInfo & info) {
			std::stringstream ss;
			ss << info.f;

			auto filename = StringUtility::replace(filenameTemplate, "${F}", ss.str());

			std::string filepath = directory + filename;
			return filepath;
		}
	};

}

#endif