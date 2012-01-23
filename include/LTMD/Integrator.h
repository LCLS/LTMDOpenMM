#ifndef OPENMM_LTMD_INTEGRATOR_H_
#define OPENMM_LTMD_INTEGRATOR_H_

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

#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include "openmm/internal/windowsExport.h"

#include "LTMD/Parameters.h"

namespace OpenMM {
	namespace LTMD {
		class Analysis;

		class OPENMM_EXPORT Integrator : public OpenMM::Integrator {
			public:
				/**
				 * Create a NMLIntegrator.
				 *
				 * @param temperature    the temperature of the heat bath (in Kelvin)
				 * @param frictionCoeff  the friction coefficient which couples the system to the heat bath
				 * @param stepSize       the step size with which to integrator the system (in picoseconds)
				 */
				Integrator( const double temperature, const double frictionCoeff, const double stepSize, const Parameters &param );

				~Integrator();

				/**
				 * Get the temperature of the heat bath (in Kelvin).
				 */
				double getTemperature() const {
					return temperature;
				}
				/**
				 * Set the temperature of the heat bath (in Kelvin).
				 */
				void setTemperature( double temp ) {
					temperature = temp;
				}
				/**
				 * Get the friction coefficient which determines how strongly the system is coupled to
				 * the heat bath.
				 */
				double getFriction() const {
					return friction;
				}
				/**
				 * Set the friction coefficient which determines how strongly the system is coupled to
				 * the heat bath.
				 */
				void setFriction( double coeff ) {
					friction = coeff;
				}

				unsigned int getNumProjectionVectors() const {
					return eigenvectors.size();
				}

				double getMinimumLimit() const {
					return minimumLimit;
				}

				void setMinimumLimit( double limit ) {
					minimumLimit = limit;
				}

				bool getProjVecChanged() const {
					return eigVecChanged;
				}

				const std::vector<std::vector<OpenMM::Vec3> >& getProjectionVectors() const {
					return eigenvectors;
				}

				void setProjectionVectors( const std::vector<std::vector<OpenMM::Vec3> >& vectors ) {
					eigenvectors = vectors;
					eigVecChanged = true;
					stepsSinceDiagonalize = 0;
				}

				double getMaxEigenvalue() const {
					return maxEigenvalue;
				}

				/**
				 * Get the random number seed.  See setRandomNumberSeed() for details.
				 */
				int getRandomNumberSeed() const {
					return randomNumberSeed;
				}

				/**
				 * Set the random number seed.  The precise meaning of this parameter is undefined, and is left up
				 * to each Platform to interpret in an appropriate way.  It is guaranteed that if two simulations
				 * are run with different random number seeds, the sequence of random forces will be different.  On
				 * the other hand, no guarantees are made about the behavior of simulations that use the same seed.
				 * In particular, Platforms are permitted to use non-deterministic algorithms which produce different
				 * results on successive runs, even if those runs were initialized identically.
				 */
				void setRandomNumberSeed( int seed ) {
					randomNumberSeed = seed;
				}

				/**
				 * Advance a simulation through time by taking a series of time steps.
				 *
				 * @param steps   the number of time steps to take
				 */
				void step( int steps = 1 );
				
				unsigned int CompletedSteps() const;

				//Minimizer
				unsigned int minimize( const unsigned int maxsteps );

			protected:
				void initialize( OpenMM::ContextImpl &context );
				std::vector<std::string> getKernelNames();
			private:
				bool DoStep();
				void DiagonalizeMinimize();
				
				// Kernel Functions
				void IntegrateStep();
				void TimeAndCounterStep();

				double LinearMinimize( const double energy );
				double QuadraticMinimize( const double energy );

				void SaveStep();
				void RevertStep();
			private:
				unsigned int mLastCompleted;
				void computeProjectionVectors();
				double maxEigenvalue;
				std::vector<std::vector<OpenMM::Vec3> > eigenvectors;
				bool eigVecChanged;
				double minimumLimit;
				double temperature, friction;
				int stepsSinceDiagonalize, randomNumberSeed;
				OpenMM::ContextImpl *context;
				OpenMM::Kernel kernel;
				const Parameters &mParameters;
				Analysis *mAnalysis;
		};
	}
}

#endif // OPENMM_LTMD_INTEGRATOR_H_
