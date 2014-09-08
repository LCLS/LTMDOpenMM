#ifndef OPENMM_LTMD_REFERENCE_STEPKERNEL_H_
#define OPENMM_LTMD_REFERENCE_STEPKERNEL_H_

#include "LTMD/StepKernel.h"

#include "ReferencePlatform.h"
#include "RealVec.h"

static std::vector<OpenMM::RealVec> &extractVelocities( OpenMM::ContextImpl &context ) {
	OpenMM::ReferencePlatform::PlatformData *data = reinterpret_cast<OpenMM::ReferencePlatform::PlatformData *>( context.getPlatformData() );
	return *( ( std::vector<OpenMM::RealVec> * ) data->velocities );
}

static std::vector<OpenMM::RealVec> &extractForces( OpenMM::ContextImpl &context ) {
	OpenMM::ReferencePlatform::PlatformData *data = reinterpret_cast<OpenMM::ReferencePlatform::PlatformData *>( context.getPlatformData() );
	return *( ( std::vector<OpenMM::RealVec> * ) data->forces );
}

static double computeShiftedKineticEnergy( OpenMM::ContextImpl &context, std::vector<double> &masses, double timeShift ) {
	std::vector<OpenMM::RealVec> &velData = extractVelocities( context );
	std::vector<OpenMM::RealVec> &forceData = extractForces( context );
	double energy = 0.0;
	for( int i = 0; i < ( int ) masses.size(); ++i ) {
		if( masses[i] > 0 ) {
			OpenMM::RealVec v = velData[i] + forceData[i] * ( timeShift / masses[i] );
			energy += masses[i] * ( v.dot( v ) );
		}
	}
	return 0.5 * energy;
}


namespace OpenMM {
	namespace LTMD {
		namespace Reference {
			typedef std::vector<double> DoubleArray;
			typedef std::vector<OpenMM::RealVec> VectorArray;

			class StepKernel : public LTMD::StepKernel {
				public:
					StepKernel( std::string name, const OpenMM::Platform &platform, OpenMM::ReferencePlatform::PlatformData &data ) : LTMD::StepKernel( name, platform ),
						data( data ) {
					}
					~StepKernel();

					/**
					 * Initialize the kernel, setting up the particle masses.
					 *
					 * @param system     the System this kernel will be applied to
					 * @param integrator the NMLIntegrator this kernel will be used for
					 */
					void initialize( const OpenMM::System &system, const Integrator &integrator );

					void Integrate( OpenMM::ContextImpl &context, const Integrator &integrator );
					void UpdateTime( const Integrator &integrator );

					void AcceptStep( OpenMM::ContextImpl &context );
					void RejectStep( OpenMM::ContextImpl &context );

					void LinearMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy );
					double QuadraticMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy );
					void updateState( OpenMM::ContextImpl &context ) {}
					virtual double computeKineticEnergy( OpenMM::ContextImpl &context, const Integrator &integrator ) {
						return computeShiftedKineticEnergy( context, mMasses, 0.5 * integrator.getStepSize() );
					}


				private:
					void Project( const Integrator &integrator, const VectorArray &in, VectorArray &out, const DoubleArray &scale, const DoubleArray &inverseScale, const bool compliment );
				private:
					unsigned int mParticles;
					double mPreviousEnergy, mMinimizerScale;
					DoubleArray mMasses, mInverseMasses, mProjectionVectors;
					VectorArray mPreviousPositions, mXPrime;
					OpenMM::ReferencePlatform::PlatformData &data;
			};
		}
	}
}

#endif // OPENMM_LTMD_REFERENCE_STEPKERNEL_H_
