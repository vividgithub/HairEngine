
#pragma once

#ifdef HAIRENGINE_ENABLE_VPBRT

#include "../precompiled/precompiled.h"
#include "../util/eigenutil.h"
#include "visualizer.h"
#include "VPly/vply.h"

namespace HairEngine {

	class HairVisualizer : public Visualizer {

	HairEngine_Public:
		HairVisualizer(const std::string & directory, const std::string & filenameTemplate):
			Visualizer(directory, filenameTemplate) {}

		void visualize(std::ostream& os, Hair& hair, const IntegrationInfo& info) override {
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
						VPly::VPlyIntAttr("si", static_cast<int32_t>(pPtr->strandPtr->index))
					);
				}
				lineStrip.stream(os);
			}
		}

	HairEngine_Protected:
		std::string directory;
		std::string filenameTemplate;
	};

}

#endif

