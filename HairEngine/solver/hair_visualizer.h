
#pragma once

#ifdef HAIRENGINE_ENABLE_VPBRT

#include "../precompiled/precompiled.h"
#include "../util/eigenutil.h"
#include "visualizer.h"
#include "VPly/vply.h"

namespace HairEngine {

	/**
	 * The interface for writing addtional virtual particle to 
	 * the HairVisualization
	 */
	class HairVisualizerVirtualParticleVisualizationInterface {
	HairEngine_Public:
		virtual size_t virtualParticleSize() const = 0;
		virtual const Hair::Particle & getVirtualParticle(size_t index) const = 0;
	};

	/**
	 * HairVisualizer is used to do the visualization of hair structure.
	 * HairVisualizer will write .vply file or .hair file based on the fileTemplate 
	 * suffix
	 */
	class HairVisualizer : public Visualizer {

		using VPI = HairVisualizerVirtualParticleVisualizationInterface;

	HairEngine_Public:
		/**
		 * Constructor
		 * 
		 * @param directory The directory that writes to
		 * @param filenameTemplate The filenameTemplate described in Visualizer class
		 * @param vpi The additional virtual particle visualization interface
		 */
		HairVisualizer(const std::string & directory, const std::string & filenameTemplate, const VPI * vpi = nullptr):
			Visualizer(directory, filenameTemplate), vpi(vpi) {
			if (StringUtility::endswith(filenameTemplate, ".hair"))
				writeVPly = false;
			else if (StringUtility::endswith(filenameTemplate, ".vply"))
				writeVPly = true;
			else
				assert(false); // Suffix invalid
		}

		void visualize(std::ostream& os, Hair& hair, const IntegrationInfo& info) override {
			if (writeVPly)
				visualizeVPly(os, hair, info);
			else
				visualizeHair(os, hair, info);
		}

	HairEngine_Protected:
		const VPI * vpi; ///< The VirtualParticleVisualizationInterface pointer
		bool writeVPly; ///< True to write .vply in the visualization, false to write .hair file

		/**
		 * Write in .hair file format
		 */
		void visualizeHair(std::ostream& os, Hair& hair, const IntegrationInfo& info) {
			hair.stream(os);
		}

		/**
		 * Write in .vply file format
		 */
		void visualizeVPly(std::ostream& os, Hair& hair, const IntegrationInfo& info) {
			for (auto sPtr = hair.strands; sPtr != hair.strandEnd(); ++sPtr) {
				// Create a VPlyAttributedLineStrip
				VPly::AttributedLineStrip lineStrip(4, VPly::VPlyIntAttr("np", sPtr->particleInfo.nparticle));
				for (auto pPtr = sPtr->particleInfo.beginPtr; pPtr != sPtr->particleInfo.endPtr; ++pPtr) {
					lineStrip.addPoint(
						EigenUtility::toVPlyVector3f(pPtr->pos),
						VPly::VPlyVector3fAttr("rpos", EigenUtility::toVPlyVector3f(pPtr->restPos)),
						VPly::VPlyVector3fAttr("vel", EigenUtility::toVPlyVector3f(pPtr->vel)),
						VPly::VPlyVector3fAttr("im", EigenUtility::toVPlyVector3f(pPtr->impulse)),
						VPly::VPlyIntAttr("gi", static_cast<int32_t>(pPtr->globalIndex)),
						VPly::VPlyIntAttr("li", static_cast<int32_t>(pPtr->localIndex)),
						VPly::VPlyIntAttr("si", static_cast<int32_t>(pPtr->strandIndex)),
						VPly::VPlyIntAttr("type", 0) // Means normal particle
					);
				}
				lineStrip.stream(os);
			}

			// Write the additional virtual particle
			if (vpi) {
				size_t nvirtual = vpi->virtualParticleSize();
				for (size_t i = 0; i < nvirtual; ++i) {
					const auto & p = vpi->getVirtualParticle(i);
					VPly::writePoint(os,
						EigenUtility::toVPlyVector3f(p.pos),
						VPly::VPlyVector3fAttr("rpos", EigenUtility::toVPlyVector3f(p.restPos)),
						VPly::VPlyVector3fAttr("vel", EigenUtility::toVPlyVector3f(p.vel)),
						VPly::VPlyVector3fAttr("im", EigenUtility::toVPlyVector3f(p.impulse)),
						VPly::VPlyIntAttr("gi", static_cast<int32_t>(p.globalIndex)),
						VPly::VPlyIntAttr("li", static_cast<int32_t>(p.localIndex)),
						VPly::VPlyIntAttr("si", static_cast<int32_t>(p.strandIndex)),
						VPly::VPlyIntAttr("type", 1) // Means virtual particle
					);
				}
			}
		}
	};

}

#endif

