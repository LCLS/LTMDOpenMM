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

#include "CudaLTMDKernelSources.h"
#include "CudaIntegrationUtilities.h"
#include "CudaContext.h"
#include "CudaArray.h"
#include <stdio.h>
#include <cuda.h>
#include <vector_functions.h>
#include <cstdlib>
#include <string>
#include <iostream>

using std::cout;
using std::endl;
#include <stdio.h>

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
			cout << "A" << endl;
			context = &contextRef;
			cout << "B" << endl;
			if( context->getSystem().getNumConstraints() > 0 ) {
				throw OpenMMException( "LTMD Integrator does not support constraints" );
			}
			cout << "C: " << StepKernel::Name() << endl;
			kernel = context->getPlatform().createKernel( StepKernel::Name(), contextRef );
			cout << "D: " << endl;
			cout << "DD" << endl;
			cout << kernel.getImpl().getName() << endl;
			((StepKernel &)( kernel.getImpl() )).initialize( contextRef.getSystem(), *this );
			//(dynamic_cast<StepKernel &>( kernel.getImpl() )).initialize( contextRef.getSystem(), *this );
			cout << "E" << endl;
		}

		std::vector<std::string> Integrator::getKernelNames() {
			std::vector<std::string> names;
			names.push_back( StepKernel::Name() );
			return names;
		}

		void Integrator::SetProjectionChanged( bool value ){
			eigVecChanged = value;
		}

		void Integrator::step( int steps ) {
			timeval start, end;
			gettimeofday( &start, 0 );

			mSimpleMinimizations = 0;
			mQuadraticMinimizations = 0;

			for( mLastCompleted = 0; mLastCompleted < steps; ++mLastCompleted ) {
				if( DoStep() == false ) break;
			}

			// Update Time
			context->setTime( context->getTime() + getStepSize() * mLastCompleted );

			// Print Minimizations
			const unsigned int total = mSimpleMinimizations + mQuadraticMinimizations;

			const double averageSimple = (double)mSimpleMinimizations / (double)mLastCompleted;
			const double averageQuadratic = (double)mQuadraticMinimizations / (double)mLastCompleted;
			const double averageTotal = (double)total / (double)mLastCompleted;

			std::cout << "[OpenMM::Minimize] " << total << " total minimizations( "
						<< mSimpleMinimizations << " simple, " << mQuadraticMinimizations << " quadratic ). "
						<< averageTotal << " per-step minimizations( " << averageSimple << " simple, "
						<< averageQuadratic << " quadratic ). Steps: " << mLastCompleted << std::endl;

			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[Integrator] Total dynamics: " << elapsed << "ms" << std::endl;
		}

				double Integrator::computeKineticEnergy() {
			return ((StepKernel &)( kernel.getImpl() )).computeKineticEnergy( *context, *this );
				//	return kernel.getAs<OpenMM::LTMD::StepKernel>().computeKineticEnergy(*context, *this);
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
			//std::cout << "Z CALCULATING FORCES..." << std::endl;
			context->updateContextState();
			context->calcForcesAndEnergy( true, false );
			//cout << "PLATFORM: " << context->getPlatform().getName() << endl;
			//std::cout << "Z DONE CALCULATING FORCES..." << std::endl;

			IntegrateStep();
			SetProjectionChanged( false );

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

		bool Integrator::minimize( const unsigned int upperbound ){
			unsigned int simple = 0, quadratic = 0;
			Minimize( upperbound, simple, quadratic );

			return (( simple + quadratic ) < upperbound);
		}

		bool Integrator::minimize( const unsigned int upperbound, const unsigned int lowerbound ){
			unsigned int simple = 0, quadratic = 0;
			Minimize( upperbound, simple, quadratic );

			std::cout << "Minimizations: " << simple << " " << quadratic << " Bound: " << lowerbound << std::endl;

			return (( simple + quadratic ) < lowerbound);
		}

		void Integrator::Minimize( const unsigned int max, unsigned int& simpleSteps, unsigned int& quadraticSteps ) {
			const double eigStore = maxEigenvalue;

			if( !mParameters.ShouldProtoMolDiagonalize && eigenvectors.size() == 0 ) computeProjectionVectors();

			SaveStep();

			//std::cout << "B CALCULATING FORCES..." << std::endl;
			double initialPE = context->calcForcesAndEnergy( true, true );
			//std::cout << "B CALCULATING FORCES..." << std::endl;
			((StepKernel &)( kernel.getImpl() )).setOldPositions();

			//context->getPositions(oldPos); // I need to get old positions here 
			simpleSteps = 0;
			quadraticSteps = 0;

			for( unsigned int i = 0; i < max; i++ ){
				SetProjectionChanged( false );

				simpleSteps++;
				double currentPE = LinearMinimize( initialPE );
				if( mParameters.isAlwaysQuadratic || currentPE > initialPE ){
					quadraticSteps++;

					double lambda = 0.0;
					currentPE = QuadraticMinimize( currentPE, lambda );
					if( currentPE > initialPE ){
						std::cout << "Quadratic Minimization Failed Energy Test [" << currentPE << ", " << initialPE << "] - Forcing Rediagonalization" << std::endl;
						computeProjectionVectors();
						break;
					}else{
						if( mParameters.ShouldForceRediagOnQuadraticLambda && lambda < 1.0 / maxEigenvalue){
							std::cout << "Quadratic Minimization Failed Lambda Test [" << lambda << ", " << 1.0 / maxEigenvalue << "] - Forcing Rediagonalization" << std::endl;
							computeProjectionVectors();
							break;
						}
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
			//std::cout << "C CALCULATING FORCES..." << std::endl;
					context->calcForcesAndEnergy( true, false );
			//std::cout << "C CALCULATING FORCES..." << std::endl;

					maxEigenvalue *= 2;
				}
			}

			mSimpleMinimizations += simpleSteps;
			mQuadraticMinimizations += quadraticSteps;

			maxEigenvalue = eigStore;
		}

		void Integrator::DiagonalizeMinimize() {
			if( !mParameters.ShouldProtoMolDiagonalize ) {
				computeProjectionVectors();
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
			((StepKernel &)( kernel.getImpl() )).Integrate( *context, *this );
			//dynamic_cast<StepKernel &>( kernel.getImpl() ).Integrate( *context, *this );
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
			((StepKernel &)( kernel.getImpl() )).UpdateTime( *this );
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
			//std::cout << "LIN MIN" << std::endl;
			((StepKernel &)( kernel.getImpl() )).LinearMinimize( *context, *this, energy );
			//std::cout << "D CALCULATING FORCES..." << std::endl;
			double retVal = context->calcForcesAndEnergy( true, true );
			//std::cout << "D CALCULATING FORCES..." << std::endl;
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
			lambda = ((StepKernel &)( kernel.getImpl() )).QuadraticMinimize( *context, *this, energy );
#ifdef KERNEL_VALIDATION
			std::cout << "[OpenMM::Integrator::Minimize] Lambda: " << lambda << " Ratio: " << ( lambda / maxEigenvalue ) << std::endl;
#endif
			//std::cout << "E CALCULATING FORCES..." << std::endl;
			double retVal = context->calcForcesAndEnergy( true, true );
			//std::cout << "E CALCULATING FORCES..." << std::endl;
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
			
			((StepKernel &)( kernel.getImpl() )).AcceptStep( *context/*, oldPos*/ ); // must pass here
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
			((StepKernel &)( kernel.getImpl() )).RejectStep( *context/*, oldPos*/ ); // must pass here
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Revert Step: " << elapsed << "ms" << std::endl;
#endif
		}
	}
}
