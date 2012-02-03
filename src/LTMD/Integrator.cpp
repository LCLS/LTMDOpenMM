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

#include <sys/time.h>

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
		Integrator::Integrator( double temperature, double frictionCoeff, double stepSize, const Parameters &params )
			: stepsSinceDiagonalize( 0 ), mParameters( params ), maxEigenvalue( 4.34e5 ), mAnalysis( new Analysis ) {
			setTemperature( temperature );
			setFriction( frictionCoeff );
 			setStepSize( stepSize );
			setConstraintTolerance( 1e-4 );
			setMinimumLimit( mParameters.minLimit );
			setRandomNumberSeed( ( int ) time( 0 ) );
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
			timeval start, end;
			gettimeofday( &start, 0 );

			for( mLastCompleted = 0; mLastCompleted < steps; ++mLastCompleted ) {
				if( DoStep() == false ) break;
			}

			// Update Time
			context->setTime( context->getTime() + getStepSize() * mLastCompleted );

			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[Integrator] Total dynamics: " << elapsed << "ms" << std::endl;
		}
		
		unsigned int Integrator::CompletedSteps() const {
			return mLastCompleted;
		}

		/* Save before integration for DiagonalizeMinimize and add test to make 
			sure its not done twice */
		bool Integrator::DoStep() {
			context->updateContextState();

			if( eigenvectors.size() == 0 || stepsSinceDiagonalize % mParameters.rediagFreq == 0 ) {
				DiagonalizeMinimize();
			}

			stepsSinceDiagonalize++;

			context->calcForcesAndEnergy( true, false );

			IntegrateStep();
			eigVecChanged = false;

			if( !minimize( mParameters.MaximumMinimizationIterations ) ){
				if( mParameters.ShouldForceRediagOnMinFail ) {
					if( mParameters.ShouldProtoMolDiagonalize ) {
						return false;
					}else{
						DiagonalizeMinimize();
					}
				}
			}

			TimeAndCounterStep();

			return true;
		}

		bool Integrator::minimize( const unsigned int maxsteps ) {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			const double eigStore = maxEigenvalue;

			if( eigenvectors.size() == 0 ) {
				computeProjectionVectors();
			}

			SaveStep();

			double initialPE = context->calcForcesAndEnergy( true, true );
			
			unsigned int steps = 0, quadraticSteps = 0;
			for( steps = 1; steps <= maxsteps; steps++ ){
				eigVecChanged = false;
				
				double currentPE = LinearMinimize( initialPE );
				if( currentPE > initialPE ) {
					quadraticSteps++;

					double lambda = 0.0;
					currentPE = QuadraticMinimize( currentPE, lambda );

					// Minimization failed if lambda is less than the minimum specified lambda
					if( lambda < mParameters.MinimumLambdaValue ) {
						//RevertStep();
						//context->calcForcesAndEnergy( true, false );
						//return false;
					}
				}
				//break if satisfies end condition
				const double diff = initialPE - currentPE;
				if( diff < getMinimumLimit() && diff >= 0.0 ) {
					break;
				}
				
				if( diff > 0.0 ) {
					SaveStep();
					initialPE = currentPE;
				} else {
					RevertStep();
					context->calcForcesAndEnergy( true, false );
					
					maxEigenvalue *= 2;
				}
				
			}
			
			std::cout << "Minimization took " << steps << " linear steps and " 
				<< quadraticSteps << " quadratic steps. Totalling " 
				<< ( steps + quadraticSteps ) << " steps." << std::endl;

			maxEigenvalue = eigStore;
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Minimize: " << elapsed << "ms" << std::endl;
#endif

			// Test to see if we reached the maximum number of minimizations
			if( steps >= maxsteps ) {
				std::cout << "[OpenMM::Minimize] Maximum minimization steps reached" << std::endl;
			}
			
			return true;
		}
		
		void Integrator::DiagonalizeMinimize() {
			if( !mParameters.ShouldProtoMolDiagonalize ) {
				unsigned int iterations = mParameters.MaximumRediagonalizations;
				if( !mParameters.ShouldForceRediagOnMinFail ) iterations = 1;

				unsigned int iteration = 0;
				for( iteration = 1; iteration <= iterations; iteration++){
					computeProjectionVectors();
					if( !minimize( mParameters.MaximumMinimizationIterations) ) break;
					if( iteration > 1 ) {
						std::cout << "[OpenMM::Integrator] Force Rediagonalization" << std::endl;
					}
				}
				std::cout << "[OpenMM::Integrator] Rediagonalized " << iteration << " times" << std::endl;
			}
		}

		void Integrator::computeProjectionVectors() {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			mAnalysis->computeEigenvectorsFull( *context, mParameters );
			setProjectionVectors( mAnalysis->getEigenvectors() );
			stepsSinceDiagonalize = 0;
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Compute Projection: " << elapsed << "ms" << std::endl;
#endif
		}

		// Kernel Functions
		void Integrator::IntegrateStep() {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			dynamic_cast<StepKernel &>( kernel.getImpl() ).Integrate( *context, *this );
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Integrate Step: " << elapsed << "ms" << std::endl;
#endif
		}

		void Integrator::TimeAndCounterStep() {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			dynamic_cast<StepKernel &>( kernel.getImpl() ).UpdateTime( *this );
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] TimeAndCounter Step: " << elapsed << "ms" << std::endl;
#endif
		}

		double Integrator::LinearMinimize( const double energy ) {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			dynamic_cast<StepKernel &>( kernel.getImpl() ).LinearMinimize( *context, *this, energy );
			double retVal = context->calcForcesAndEnergy( true, true );
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Linear Minimize: " << elapsed << "ms" << std::endl;
#endif
			return retVal;
		}

		double Integrator::QuadraticMinimize( const double energy, double& lambda ) {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			lambda = dynamic_cast<StepKernel &>( kernel.getImpl() ).QuadraticMinimize( *context, *this, energy );
			lambda = std::abs( lambda );
#ifdef KERNEL_VALIDATION
			std::cout << "[OpenMM::Integrator::Minimize] Lambda: " << lambda << " Ratio: " << ( lambda / ( 1 / 5e5 ) ) << std::endl;
#endif
			double retVal = context->calcForcesAndEnergy( true, true );
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Quadratic Minimize: " << elapsed << "ms" << std::endl;
#endif
			return retVal;
		}

		void Integrator::SaveStep() {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			dynamic_cast<StepKernel &>( kernel.getImpl() ).AcceptStep( *context );
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Save Step: " << elapsed << "ms" << std::endl;
#endif
		}

		void Integrator::RevertStep() {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			dynamic_cast<StepKernel &>( kernel.getImpl() ).RejectStep( *context );
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Revert Step: " << elapsed << "ms" << std::endl;
#endif
		}
	}
}
