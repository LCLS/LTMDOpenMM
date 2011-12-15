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

#include <ctime>
#include <string>
#include <iostream>

#ifdef PROFILE_INTEGRATOR 
#include <sys/time.h>
#endif

#include "openmm/System.h"
#include "openmm/Context.h"
#include "openmm/kernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include "LTMD/Analysis.h"
#include "LTMD/Integrator.h"
#include "LTMD/StepKernel.h"

namespace OpenMM {
	namespace LTMD {
		Integrator::Integrator( double temperature, double frictionCoeff, double stepSize, Parameters *params )
			: stepsSinceDiagonalize( 0 ), mAnalysis( new Analysis ) {
			setTemperature( temperature );
			setFriction( frictionCoeff );
			setStepSize( stepSize );
			setConstraintTolerance( 1e-4 );
			setMinimumLimit( params->minLimit );
			setRandomNumberSeed( (int) time( 0 ) );
			parameters = params;
			rediagonalizeFrequency = params->rediagFreq;
		}
		
		Integrator::~Integrator() {
			delete mAnalysis;
		}

		void Integrator::initialize( ContextImpl &contextRef ) {
			context = &contextRef;
			if( context->getSystem().getNumConstraints() > 0 ) {
				throw OpenMMException( "LTMD Integrator does not support constraints" );
			}
			kernel = context->getPlatform().createKernel( StepKernel::Name(), contextRef );
			dynamic_cast<StepKernel &>( kernel.getImpl() ).initialize( contextRef.getSystem(), *this );
		}

		std::vector<std::string> Integrator::getKernelNames() {
			std::vector<std::string> names;
			names.push_back( StepKernel::Name() );
			return names;
		}

		void Integrator::step( int steps ) {
			for( unsigned int i = 0; i < steps; ++i ) DoStep();

			// Update Time
			context->setTime( context->getTime() + getStepSize() * steps );
		}
		
		void Integrator::DoStep() {
			#ifdef PROFILE_INTEGRATOR
				timeval start, end;
				gettimeofday( &start, 0 );
			#endif
			context->updateContextState();

			if( eigenvectors.size() == 0 || stepsSinceDiagonalize % rediagonalizeFrequency == 0 ) {
				computeProjectionVectors();
			}
			
			stepsSinceDiagonalize++;

			context->calcForcesAndEnergy( true, false );

			IntegrateStep();
			eigVecChanged = false;

			minimize();

			TimeAndCounterStep();
			#ifdef PROFILE_INTEGRATOR
				gettimeofday( &end, 0 );
				double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
				std::cout << "[Integrator] Step: " << elapsed << "ms" << std::endl;
			#endif
		}

		void Integrator::minimize( const unsigned int maxsteps ) {
			#ifdef PROFILE_INTEGRATOR
				timeval start, end;
				gettimeofday( &start, 0 );
			#endif 
			const double eigStore = maxEigenvalue;

			if( eigenvectors.size() == 0 ) computeProjectionVectors();

			SaveStep();

			double initialPE = context->calcForcesAndEnergy( true, true );
			for( int i = 0; i < maxsteps; ++i ) {
				eigVecChanged = false;

				double currentPE = LinearMinimize( initialPE );
				if( currentPE > initialPE ) currentPE = QuadraticMinimize( currentPE );
				
				//break if satisfies end condition
				const double diff = initialPE - currentPE;
				if( diff < getMinimumLimit() && diff >= 0.0 ) break;

				if( diff > 0.0 ){
					SaveStep();
					initialPE = currentPE;
				}else{
					RevertStep();
					context->calcForcesAndEnergy( true, false );

					maxEigenvalue *= 2;
				}
			}

			maxEigenvalue = eigStore;
			#ifdef PROFILE_INTEGRATOR
				gettimeofday( &end, 0 );
				double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
				std::cout << "[Integrator] Minimize: " << elapsed << "ms" << std::endl;
			#endif
		}

		void Integrator::computeProjectionVectors() {
			#ifdef PROFILE_INTEGRATOR
				timeval start, end;
				gettimeofday( &start, 0 );
			#endif
			mAnalysis->computeEigenvectorsFull( *context, parameters );
			setProjectionVectors( mAnalysis->getEigenvectors() );
			maxEigenvalue = mAnalysis->getMaxEigenvalue();
			stepsSinceDiagonalize = 0;
			#ifdef PROFILE_INTEGRATOR
				gettimeofday( &end, 0 );
				double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
				std::cout << "[Integrator] Compute Projection: " << elapsed << "ms" << std::endl;
			#endif
		}
		
		// Kernel Functions
		void Integrator::IntegrateStep() {
			#ifdef PROFILE_INTEGRATOR
				timeval start, end;
				gettimeofday( &start, 0 );
			#endif 
			dynamic_cast<StepKernel&>( kernel.getImpl() ).execute( *context, *this, 0.0, 1 );
			#ifdef PROFILE_INTEGRATOR
				gettimeofday( &end, 0 );
				double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
				std::cout << "[Integrator] Integrate Step: " << elapsed << "ms" << std::endl;
			#endif
		}
		
		void Integrator::TimeAndCounterStep() {
			#ifdef PROFILE_INTEGRATOR
				timeval start, end;
				gettimeofday( &start, 0 );
			#endif 
			dynamic_cast<StepKernel&>( kernel.getImpl() ).execute( *context, *this, 0.0, 2 );
			#ifdef PROFILE_INTEGRATOR
				gettimeofday( &end, 0 );
				double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
				std::cout << "[Integrator] TimeAndCounter Step: " << elapsed << "ms" << std::endl;
			#endif
		}
		
		double Integrator::LinearMinimize( const double energy ){
			#ifdef PROFILE_INTEGRATOR
				timeval start, end;
				gettimeofday( &start, 0 );
			#endif 
			dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, energy, 3 );
			double retVal = context->calcForcesAndEnergy( false, true );
			#ifdef PROFILE_INTEGRATOR
				gettimeofday( &end, 0 );
				double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
				std::cout << "[Integrator] Linear Minimize: " << elapsed << "ms" << std::endl;
			#endif
			return retVal;
		}
		
		double Integrator::QuadraticMinimize( const double energy ) {
			#ifdef PROFILE_INTEGRATOR
				timeval start, end;
				gettimeofday( &start, 0 );
			#endif 
			context->calcForcesAndEnergy( true, true );
			dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, energy, 4 );
			double retVal = context->calcForcesAndEnergy( false, true );
			#ifdef PROFILE_INTEGRATOR
				gettimeofday( &end, 0 );
				double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
				std::cout << "[Integrator] Quadratic Minimize: " << elapsed << "ms" << std::endl;
			#endif
			return retVal;
		}
		
		void Integrator::SaveStep() {
			#ifdef PROFILE_INTEGRATOR
				timeval start, end;
				gettimeofday( &start, 0 );
			#endif 
			dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, 0.0, 6 );
			#ifdef PROFILE_INTEGRATOR
				gettimeofday( &end, 0 );
				double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
				std::cout << "[Integrator] Save Step: " << elapsed << "ms" << std::endl;
			#endif
		}
		
		void Integrator::RevertStep() {
			#ifdef PROFILE_INTEGRATOR
				timeval start, end;
				gettimeofday( &start, 0 );
			#endif 
			dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, 0.0, 5 );
			#ifdef PROFILE_INTEGRATOR
				gettimeofday( &end, 0 );
				double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
				std::cout << "[Integrator] Revert Step: " << elapsed << "ms" << std::endl;
			#endif
		}
	}
}
