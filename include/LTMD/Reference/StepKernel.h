#ifndef OPENMM_LTMD_REFERENCE_STEPKERNEL_H_
#define OPENMM_LTMD_REFERENCE_STEPKERNEL_H_


/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009 Stanford University and the Authors.           *
 * Authors: Chris Sweet                                                       *
 * Contributors: Christopher Bruns                                            *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

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
