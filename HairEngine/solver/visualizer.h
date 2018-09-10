#pragma once

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
		 * @param timestep The time interval for writing another visualization file to the disk. It indicates the time
		 * interval to write a visualization summary file to the disk so that it could avoid write multiple
		 * intermediate simulation step with a small integration time. A value less or equal to 0.0f indicating
		 * to write visualization file when the "solve" is called.
		 */
		Visualizer(const std::string & directory, const std::string & filenameTemplate, float timestep=0.0f)
			: directory(directory), filenameTemplate(filenameTemplate), timestep(timestep) {

			// Append path seprator if needed
			if (!StringUtility::endswith(this->directory, StringUtility::getPathSeparator()))
				this->directory += StringUtility::getPathSeparator();
		}

		void solve(Hair& hair, const IntegrationInfo& info) final override {

			currentTime += info.t;

			// Works well even when info.t > timestep
			if (timestep <= 0.0f || currentTime >= 0.995f * timestep) {
				auto filepath = getFilepath(info);
				++indexCounter;

				std::fstream fout(filepath, std::ios::out | std::ios::binary);
				visualize(fout, hair, info);

				fout.close();

				// When timestep <= 0.0, it is safe since we don't use the currentTime
				currentTime -= timestep;
			}
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
		float timestep;

		float currentTime = 0.0f;
		int indexCounter = 1; // We use the counter starts from 1

		std::string getFilepath(const IntegrationInfo & info) {
			std::stringstream ss;
			ss << indexCounter;

			auto filename = StringUtility::replace(filenameTemplate, "${F}", ss.str());

			std::string filepath = directory + filename;
			return filepath;
		}
	};

}