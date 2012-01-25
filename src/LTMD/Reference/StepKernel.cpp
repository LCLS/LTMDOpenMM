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

#include "LTMD/Reference/StepKernel.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/internal/ContextImpl.h"
#include "SimTKReference/ReferenceCCMAAlgorithm.h"
#include "SimTKUtilities/SimTKOpenMMUtilities.h"
#include <vector>

namespace OpenMM {
	namespace LTMD {
		namespace Reference {
			static int **allocateIntArray( int length, int width ) {
				int **array = new int*[length];
				for( int i = 0; i < length; ++i ) {
					array[i] = new int[width];
				}
				return array;
			}

			static void disposeIntArray( int **array, int size ) {
				if( array ) {
					for( int i = 0; i < size; ++i ) {
						delete[] array[i];
					}
					delete[] array;
				}
			}

			static std::vector<RealVec>& extractPositions( ContextImpl &context ) {
				ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>( context.getPlatformData() );
				return *( ( std::vector<RealVec>* ) data->positions );
			}

			static std::vector<RealVec>& extractVelocities( ContextImpl &context ) {
				ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>( context.getPlatformData() );
				return *( ( std::vector<RealVec>* ) data->velocities );
			}

			static std::vector<RealVec>& extractForces( ContextImpl &context ) {
				ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>( context.getPlatformData() );
				return *( ( std::vector<RealVec>* ) data->forces );
			}

			static void findAnglesForCCMA( const System &system, std::vector<ReferenceCCMAAlgorithm::AngleInfo>& angles ) {
				for( int i = 0; i < system.getNumForces(); i++ ) {
					const HarmonicAngleForce *force = dynamic_cast<const HarmonicAngleForce *>( &system.getForce( i ) );
					if( force != NULL ) {
						for( int j = 0; j < force->getNumAngles(); j++ ) {
							int atom1, atom2, atom3;
							double angle, k;
							force->getAngleParameters( j, atom1, atom2, atom3, angle, k );
							angles.push_back( ReferenceCCMAAlgorithm::AngleInfo( atom1, atom2, atom3, ( RealOpenMM )angle ) );
						}
					}
				}
			}


			StepKernel::~StepKernel() {
				if( dynamics ) {
					delete dynamics;
				}
				if( constraints ) {
					delete constraints;
				}
				if( constraintIndices ) {
					disposeIntArray( constraintIndices, numConstraints );
				}
				if( constraintDistances ) {
					delete[] constraintDistances;
				}
				if( projectionVectors ) {
					delete projectionVectors;
				}
			}

			void StepKernel::initialize( const System &system, const Integrator &integrator ) {
				int numParticles = system.getNumParticles();
				masses.resize( numParticles );
				for( int i = 0; i < numParticles; ++i ) {
					masses[i] = static_cast<RealOpenMM>( system.getParticleMass( i ) );
				}
				numConstraints = system.getNumConstraints();
				constraintIndices = allocateIntArray( numConstraints, 2 );
				constraintDistances = new RealOpenMM[numConstraints];
				for( int i = 0; i < numConstraints; ++i ) {
					int particle1, particle2;
					double distance;
					system.getConstraintParameters( i, particle1, particle2, distance );
					constraintIndices[i][0] = particle1;
					constraintIndices[i][1] = particle2;
					constraintDistances[i] = static_cast<RealOpenMM>( distance );
				}
				SimTKOpenMMUtilities::setRandomNumberSeed( ( unsigned int ) integrator.getRandomNumberSeed() );
			}

			void StepKernel::execute( ContextImpl &context, const Integrator &integrator, const double currentPE, const int stepType ) {

				double temperature = integrator.getTemperature();
				double friction = integrator.getFriction();
				double stepSize = integrator.getStepSize();
				const std::vector<std::vector<Vec3> >& dProjectionVectors = integrator.getProjectionVectors();
				unsigned int numProjectionVectors = integrator.getNumProjectionVectors();
				bool projVecChanged = integrator.getProjVecChanged();
				double minimumLimit = integrator.getMinimumLimit();
				double maxEig = integrator.getMaxEigenvalue();


				std::vector<RealVec>& posData = extractPositions( context );
				std::vector<RealVec>& velData = extractVelocities( context );
				std::vector<RealVec>& forceData = extractForces( context );

				//projection vectors
				if( projectionVectors == 0 || projVecChanged ) {
					unsigned int arraySz = numProjectionVectors * context.getSystem().getNumParticles() * 3;
					if( projectionVectors == 0 ) {
						projectionVectors = new RealOpenMM[arraySz];
					}
					int index = 0;
					for( int i = 0; i < ( int ) dProjectionVectors.size(); i++ )
						for( int j = 0; j < ( int ) dProjectionVectors[i].size(); j++ )
							for( int k = 0; k < 3; k++ ) {
								projectionVectors[index++] = static_cast<RealOpenMM>( dProjectionVectors[i][j][k] );
							}
				}

				if( dynamics == 0 || temperature != prevTemp || friction != prevFriction || stepSize != prevStepSize ) {
					// Recreate the computation objects with the new parameters.

					if( dynamics ) {
						delete dynamics;
						delete constraints;
					}
					RealOpenMM tau = static_cast<RealOpenMM>( friction == 0.0 ? 0.0 : 1.0 / friction );

					dynamics = new Dynamics( context.getSystem().getNumParticles(),
											 static_cast<RealOpenMM>( stepSize ),
											 static_cast<RealOpenMM>( tau ),

											 static_cast<RealOpenMM>( temperature ),
											 projectionVectors,
											 numProjectionVectors,
											 static_cast<RealOpenMM>( maxEig ) );

					std::vector<ReferenceCCMAAlgorithm::AngleInfo> angles;
					findAnglesForCCMA( context.getSystem(), angles );
					constraints = new ReferenceCCMAAlgorithm( context.getSystem().getNumParticles(), numConstraints, constraintIndices, constraintDistances, masses, angles, ( RealOpenMM )integrator.getConstraintTolerance() );
					dynamics->setReferenceConstraintAlgorithm( constraints );
					prevTemp = temperature;
					prevFriction = friction;
					prevStepSize = stepSize;
				}
				dynamics->SetMaxEigenValue( maxEig );
				dynamics->update( posData, velData, forceData, masses, currentPE, stepType );
				//update at dynamic step 2
				if( stepType == 2 ) {
					data.time += stepSize;
					data.stepCount++;
				}

			}
		}
	}
}

