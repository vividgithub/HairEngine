
#pragma once

#ifdef HAIRENGINE_ENABLE_VPBRT

#include "../precompiled/precompiled.h"
#include "visualizer.h"
#include "VPly/vply.h"

namespace HairEngine {

	class HairParticleVisualizer : public Visualizer {

	HairEngine_Public:
		HairParticleVisualizer(const std::string & directory, const std::string & filenameTemplate):
			Visualizer(directory, filenameTemplate) {}

		void visualize(std::ostream& os, Hair& hair, const IntegrationInfo& info) override {
			for (auto p = hair.particles; p != hair.particleEnd(); ++p) {
				VPly::writePoint(
					os,
					EigenUtility::toVPlyVector3f(p->pos),
					VPly::VPlyVector3fAttr("restpos", EigenUtility::toVPlyVector3f(p->restPos)),
					VPly::VPlyVector3fAttr("vel", EigenUtility::toVPlyVector3f(p->vel)),
					VPly::VPlyVector3fAttr("im", EigenUtility::toVPlyVector3f(p->impulse)),
					VPly::VPlyIntAttr("li", static_cast<int32_t>(p->localIndex)),
					VPly::VPlyIntAttr("gi", static_cast<int32_t>(p->globalIndex)),
					VPly::VPlyIntAttr("si", static_cast<int32_t>(p->strandPtr->index))
				);
			}
		}

	HairEngine_Protected:
		std::string directory;
		std::string filenameTemplate;
	};

}

#endif

