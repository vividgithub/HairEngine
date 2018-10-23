//
// Created by vivi on 2018/10/22.
//

#pragma once

#include "../util/parmutil.h"

namespace HairEngine {
	struct SelleMassSpringConfiguration {
		float stretchStiffness; ///< Stiffness of the stretch spring
		float bendingStiffness; ///< Stiffness of the bending spring
		float torsionStiffness; ///< Stiffness of the torsion spring
		float altitudeStiffness; ///< Stiffness of the altitude spring
		float damping; ///< The damping coefficient

		/// Rigidness defines how rigid of a strand is.
		/// 1.0 means the strand will behave as rigid as possible, and 0.0 means the strand will be completely driven
		/// by strand dynamics equations. Since normally hair will be more rigid in hair root and less rigid in the
		/// tip, the parameter is passed as "VaryingFloat" which x = 0.0 defines the rigidness for the hair root
		/// and x = 1.0 defines the rigidness for the tip
		VaryingFloat rigidness;

		/// If the segment's length is larger than "strainLimitingLengthTolerance * rest_length", then the
		/// position will be fixed by strain limiting, a value less or equal to 1.0f will disable the strain limiting
		float strainLimitingLengthTolerance;

		float colinearMaxDegree; ///< We will insert additional virtual particles if two adjacent line segments are "nearly" colinear, we treat the two adjacent line segment colinear is the included angle is less than colinearMaxDegree
		float mass; ///< The mass of the single hair strand

		/// We allow to specify a detailed integration time to guarantee stability. So that if the timestep is too
		/// large, then it will be splitted into several integration.
		float maxIntegrationTime;

		/**
		 * Constructor
		 */
		SelleMassSpringConfiguration(
				float stretchStiffness,
				float bendingStiffness,
				float torsionStiffness,
				float altitudeStiffness,
				float damping,
				VaryingFloat rigidness,
				float strainLimitingLengthTolerance,
				float colinearMaxDegree,
				float mass,
				float maxIntegrationTime
		):
				stretchStiffness(stretchStiffness),
				bendingStiffness(bendingStiffness),
				torsionStiffness(torsionStiffness),
				altitudeStiffness(altitudeStiffness),
				damping(damping),
				rigidness(std::move(rigidness)),
				strainLimitingLengthTolerance(strainLimitingLengthTolerance),
				colinearMaxDegree(colinearMaxDegree),
				mass(mass),
				maxIntegrationTime(maxIntegrationTime) {}
	};
}
