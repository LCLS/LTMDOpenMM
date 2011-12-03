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

#include "openmm/System.h"
#include "openmm/Context.h"
#include "openmm/kernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include "LTMD/Analysis.h"
#include "LTMD/Parameters.h"
#include "LTMD/Integrator.h"
#include "LTMD/StepKernel.h"

namespace OpenMM {
	namespace LTMD {
		Integrator::Integrator( double temperature, double frictionCoeff, double stepSize, Parameters *params )
			: stepsSinceDiagonalize( 0 ), rediagonalizeFrequency( 1000 ) {
			setTemperature( temperature );
			setFriction( frictionCoeff );
			setStepSize( stepSize );
			setConstraintTolerance( 1e-4 );
			setMinimumLimit( 0.1 );
			setRandomNumberSeed( ( int ) time( NULL ) );
			parameters = params;
		}

		void Integrator::initialize( ContextImpl &contextRef ) {
			context = &contextRef;
			if( context->getSystem().getNumConstraints() > 0 ) {
				throw OpenMMException( "NMLIntegrator does not support systems with constraints" );
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
			for( int i = 0; i < steps; ++i ) {
				context->updateContextState();

				if( eigenvectors.size() == 0 || stepsSinceDiagonalize % rediagonalizeFrequency == 0 ) {
					computeProjectionVectors();
				}
				stepsSinceDiagonalize++;

				context->calcForcesAndEnergy( true, false );

				// Integrate one step
				dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, 0.0, 1 );
				//in case projection vectors changed, clear flag
				eigVecChanged = false;

				//minimize compliment space, set maximum minimizer loops to 50
				minimize( 50 );

				// Update the time and step counter.
				dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, 0.0, 2 );
			}

			//update time
			context->setTime( context->getTime() + getStepSize() * steps );

		}

		void Integrator::minimize( int maxsteps ) {
			const double eigStore = maxEigenvalue;

			if( eigenvectors.size() == 0 ) computeProjectionVectors();

			//minimum limit
			const double minlim = getMinimumLimit();

			// Record initial positions.
			dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, 0.0, 6 );

			double initialPE = context->calcForcesAndEnergy( true, true );
			for( int i = 0; i < maxsteps; ++i ) {
				dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, initialPE, 3 ); //stepType 3 is simple minimizer
				eigVecChanged = false;

				double currentPE = context->calcForcesAndEnergy( true, true );
				if( currentPE > initialPE ) {
					std::cout << "Quadratic Minimize on step " << i << std::endl;
					dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, currentPE, 4 ); //stepType 4 is quadratic minimizer
					currentPE = context->calcForcesAndEnergy( true, true );

				}

				//break if satisfies end condition
				const double diff = initialPE - currentPE;
				if( diff < minlim && diff >= 0.0 ) {
					std::cout << "Minimisation finishes in " << i << " steps." << std::endl;
					break;
				}

				// Accept or reject the step
				dynamic_cast<StepKernel &>( kernel.getImpl() ).execute( *context, *this, currentPE, diff < 0.0 ? 5 : 6 );
				if( diff < 0.0 ) {
					context->calcForcesAndEnergy( true, false );

					maxEigenvalue *= 2;
					std::cout << "Minimize failed, maxEigenvalue now " << maxEigenvalue << "." << std::endl;
				} else {
					initialPE = currentPE;
				}
			}

			maxEigenvalue = eigStore;
		}

		void Integrator::computeProjectionVectors() {
			Analysis nma;
			nma.computeEigenvectorsFull( *context, parameters );
			const std::vector<std::vector<Vec3> > e1 = nma.getEigenvectors();
			setProjectionVectors( nma.getEigenvectors() );
			maxEigenvalue = nma.getMaxEigenvalue();
			stepsSinceDiagonalize = 0;
		}
	}
}
