//
// Created by vivi on 16/06/2018.
//

#include "hair.h"

namespace HairEngine {

	std::ostream & operator<<(std::ostream & os, const Hair::Particle & p) {
		os << "Particle[restPos=" << EigenUtility::toString(p.restPos)
		          << ",pos=" << EigenUtility::toString(p.pos) << ",vel=" << EigenUtility::toString(p.vel)
		          << ",impluse=" << EigenUtility::toString(p.impulse) << ",localIndex=" << p.localIndex
		          << ",globalIndex=" << p.globalIndex << ']';
		return os;
	}

	std::ostream & operator<<(std::ostream & os, const Hair::Segment & s) {
		os << "Segment[p1=" << (*s.p1) << ",p2=" << *(s.p2)
		   << ",localIndex=" << s.localIndex << ",globalIndex=" << s.globalIndex << "]";
		return os;
	}

	std::ostream & operator<<(std::ostream & os, const Hair::Strand & strand) {
		os << "Strand[index=" << strand.index
		   << ",particleCount=" << strand.particlePtrs.size()
		   << ",segmentSize=" << strand.segmentPtrs.size();

		for (size_t i = 0; i < strand.particlePtrs.size(); ++i) {
			os << ",p" << i << '=' << *strand.particlePtrs[i];
		}
		for (size_t i = 0; i < strand.segmentPtrs.size(); ++i) {
			os << ",s" << i << '=' << *strand.segmentPtrs[i];
		}

		os << ']';

		return os;
	}

	std::ostream & operator<<(std::ostream & os, const Hair & hair) {
		constexpr char indent[] = "    ";

		os << "Hair[\n";

		os << indent << "Particle=[\n";
		for (size_t i = 0; i < hair.particles.size(); ++i)
			os << indent << indent << hair.particles[i] << "\n";
		os << indent << "]\n";

		os << indent << "Segments=[\n";
		for (size_t i = 0; i < hair.segments.size(); ++i)
			os << indent << indent << hair.segments[i] << "\n";
		os << indent << "]\n";

		os << indent << "Strands=[";
		for (size_t i = 0; i < hair.strands.size(); ++i)
			os << indent << indent << hair.strands[i] << "\n";
		os << indent << "]\n";

		os << "]\n";

		return os;
	}
}