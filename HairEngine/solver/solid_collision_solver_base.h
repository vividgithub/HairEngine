#pragma once

#include "integration_info.h"
#include "solver.h"

namespace HairEngine {

	/**
	 * The base class of solid collision 
	 */
	class SolidCollisionSolverBase: public Solver {

	HairEngine_Protected:

		static constexpr const float DISTANCE_INFINITY = 1e30f;

	HairEngine_Public:

		struct Configuration {
			float relativeContour;
			float fraction;
			float degenerationFactor;

			Configuration(float relativeContour, float fraction, float degenerationFactor = 1.0f + 5e-3f):
				relativeContour(relativeContour), fraction(fraction), degenerationFactor(degenerationFactor) {}
		};

		/**
		 * Constructor
		 * 
		 * Initaliza a collision object with two transform, the previousTransform could be used to caculate the velocity of a 
		 * rigid collision object at the initial state.
		 */
		explicit SolidCollisionSolverBase(
			const Eigen::Affine3f & transform,
			const Eigen::Affine3f & previousTransform,
			float t,
			const Configuration &conf
		): conf(conf) {
			updateTransform(transform, t);
			previousLocalToWorldTransform = previousTransform;
		}

		/**
		 * Constructor
		 * 
		 * An convenient initalizer which assume that the collsion object is static at the initial state
		 */
		explicit SolidCollisionSolverBase(
			const Eigen::Affine3f & transform,
			const Configuration & conf
		) : SolidCollisionSolverBase(transform, transform, 1.0f, conf) {}

		/**
		 * Constructor
		 * 
		 * We provide a normal solution for transforming the modelDistance and its gradient direction to the world
		 * coordinate. The input worldPos indicates the query world position for input, the outGradientPtr specifies
		 * the gradient direction (we don't ensure the magnitude is right and it has been normalized)
		 */
		virtual float distance(const Eigen::Vector3f & worldPos, Eigen::Vector3f *outGradientPtr = nullptr) const {
			Eigen::Vector3f gradient;

			const float distanceInModelSpace = modelDistance(worldToLocalTransform * worldPos, &gradient);

			if (distanceInModelSpace == DISTANCE_INFINITY)
				return distanceInModelSpace;

			float sign = (distanceInModelSpace < 0.0f ? -1.0f : 1.0f);

			//If the gradient at that space is zero, we specify gradient to the x coordinate
			if (gradient == Eigen::Vector3f::Zero())
				gradient = { 1.0f, 0.0f, 0.0f };

			//Rescale the length of gradient to the distance in model space and transform it to the world space
			gradient *= std::abs(distanceInModelSpace) / gradient.norm();
			gradient = localToWorldLinear * gradient;

			if (outGradientPtr)
				*outGradientPtr = gradient;

			//We ensure that the transform will not swap handness, thus the model will not flip the outside to the inside,
			//and inside to the outside. So the sign will not be inverse.
			return sign * (gradient.norm());
		}

		void solve(Hair& hair, const IntegrationInfo& info) override {

			float isoContour = conf.relativeContour * boundingBox().diagonal();

			mapParticle(true, [this, &info, isoContour](Hair::Particle::Ptr par) {

				Eigen::Vector3f outGradient;
				Eigen::Vector3f newPosition = par->predictedPos(info.t);

				float signedDistance = distance(newPosition, &outGradient);

				//If it is outside the isoContour
				if (signedDistance > isoContour) {
					par->pos = newPosition;
					return;
				}

				outGradient.normalize();

				//Velocity for normal and tangent for the particles
				Eigen::Vector3f vpt, vpn;
				//Position for normal and tangent for the collision object
				Eigen::Vector3f v, vt, vn;
				Eigen::Vector3f & vp = par->vel;

				v = velocity(newPosition);
				MathUtility::projection(vp, outGradient, vpn, vpt);
				MathUtility::projection(v, outGradient, vn, vt);

				//Update the tangent velocity
				Eigen::Vector3f vrelt = vpt - vt;
				vrelt = std::max<float>(0.0f, 1.0f - conf.fraction * (vpn - vn).norm() / vrelt.norm()) * vrelt;
				vpt = vt + vrelt;

				//Update the normal velocity
				vpn = vn;

				par->vel = vpt + vpn;
				par->pos += par->vel * info.t;

				//If the new position is in the inside the body, then push it again
				signedDistance = distance(par->pos, &outGradient);
				if (signedDistance <= isoContour)
					par->pos += ((isoContour - signedDistance) * conf.degenerationFactor) * outGradient.normalized();

			});
		}

		/**
		 * Get the local to world transform
		 * 
		 * @return A Affine3f indicating the local to world transform
		 */
		const Eigen::Affine3f &getLocalToWorldTransform() const {
			return localToWorldTransform;
		}

		/**
		 * Get the world to local transform
		 * 
		 * @return A Affine3f indicating the world to local transform
		 */
		const Eigen::Affine3f &getWorldToLocalTransform() const {
			return worldToLocalTransform;
		}

		/**
		 * Get the velocity for the collision object for a given worldPos.
		 * It won't check whether the worldPos is in the bounding box of not.
		 * 
		 * @param worldPos The world position
		 * @return The velocity
		 */
		Eigen::Vector3f velocity(const Eigen::Vector3f & worldPos) const {
			Eigen::Vector3f modelPos = worldToLocalTransform * worldPos;
			Eigen::Vector3f previousWorldPos = previousLocalToWorldTransform * modelPos;
			return (1.0f / deltaTime) * (worldPos - previousWorldPos);
		}

		/**
		 * The axis-aligned bounding box of the collision object, derived class should implement it
		 * 
		 * @return The bounding box
		 */
		virtual Eigen::AlignedBox3f boundingBox() const = 0;

		Eigen::Vector3f worldDiagnoal() const {
			Eigen::AlignedBox3f bb = boundingBox();
			return localToWorldTransform * (bb.max() - bb.min());
		}

		/**
		 * Return the distance for a specific query position (in model space) and return a SIGNED distance with a optional
		 * gradient (we don't ensure the magnitude is right and it has been normalized), the graident points out a direction
		 * indicating that the modelPos is increasing fast.
		 * 
		 * @param modelPos The position in model coordinate
		 * @param outGradientPtr The gradient (in Vector3f) output
		 */
		virtual float modelDistance(const Eigen::Vector3f & modelPos, Eigen::Vector3f *outGradientPtr = nullptr) const = 0;

		/**
		 * Update the transformation, it takes the second parameter the t from the last transform to current transform,
		 * which is used to caculate the velocity.
		 * 
		 * @param transform The current transform
		 * @param t The time for the transform from previous transform to current transform 
		 */
		void updateTransform(const Eigen::Affine3f & transform, const float t) {
			deltaTime = t;

			previousLocalToWorldTransform = localToWorldTransform;

			localToWorldTransform = transform;
			worldToLocalTransform = localToWorldTransform.inverse();

			localToWorldLinear = localToWorldTransform.linear();
			localToWorldTranslation = localToWorldTransform.translation();
			worldToLocalLinear = worldToLocalTransform.linear();
			worldToLocalTranslation = worldToLocalTransform.translation();
		}

	HairEngine_Protected:

		/*
		 * Two transform to transform the world and local coordinate. We use worldToLocalTransform to transform a point
		 * in the world space to the local space for distance querying. And we use the localToWorldTransform to transform
		 * it back and handling the speed and velocities.
		 */
		Eigen::Affine3f localToWorldTransform, worldToLocalTransform;
		Eigen::Matrix3f localToWorldLinear, worldToLocalLinear;
		Eigen::Vector3f localToWorldTranslation, worldToLocalTranslation;

		//Use for caculating the velocity
		Eigen::Affine3f previousLocalToWorldTransform;

		//The time interval from the last transform to current transform
		float deltaTime;

		// Configuration
		Configuration conf;
	};
}