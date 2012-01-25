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
 * Contributors: Christopher Bruns, Pande Group                               *
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

#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include "SimTKUtilities/SimTKOpenMMCommon.h"
#include "SimTKUtilities/SimTKOpenMMLog.h"
#include "SimTKUtilities/SimTKOpenMMUtilities.h"
#include "LTMD/Reference/Dynamics.h"

namespace OpenMM {
	namespace LTMD {
		namespace Reference {

			/**---------------------------------------------------------------------------------------

			   ReferenceNMLDynamics constructor

			   @param numberOfAtoms  number of atoms
			   @param deltaT         delta t for dynamics
			   @param tau            viscosity(?)
			   @param temperature    temperature

			   --------------------------------------------------------------------------------------- */

			Dynamics::Dynamics( int numberOfAtoms,
								double deltaT, const double tau,
								double temperature,
								double *projectionVectors,
								unsigned int numProjectionVectors,
								double minimumLimit, double maxEig ) :
				ReferenceDynamics( numberOfAtoms, deltaT, temperature ), mTau( tau == 0.0 ? 1.0 : tau ),
				_projectionVectors( projectionVectors ), _numProjectionVectors( numProjectionVectors ),
				_minimumLimit( minimumLimit ), _maxEig( maxEig )  {

				// ---------------------------------------------------------------------------------------

				static const char *methodName      = "\nReferenceNMLDynamics::ReferenceNMLDynamics";

				// ---------------------------------------------------------------------------------------
				xPrime.resize( numberOfAtoms );
				oldPositions.resize( numberOfAtoms );
				inverseMasses.resize( numberOfAtoms );
			}

			Dynamics::~Dynamics( ) {

			}

			void Dynamics::SetMaxEigenValue( double value ) {
				_maxEig = value;
			}

			/**---------------------------------------------------------------------------------------

			   Update -- driver routine for performing stochastic dynamics update of coordinates
			   and velocities

			   @param numberOfAtoms       number of atoms
			   @param atomCoordinates     atom coordinates
			   @param velocities          velocities
			   @param forces              forces
			   @param masses              atom masses

			   @return ReferenceDynamics::DefaultReturn

			   --------------------------------------------------------------------------------------- */

			int Dynamics::update( int numberOfAtoms, VectorArray& atomCoordinates,
								  VectorArray& velocities,
								  VectorArray& forces, DoubleArray& masses,
								  const double currentPE, const int stepType ) {

				// ---------------------------------------------------------------------------------------

				static const char *methodName      = "\nReferenceNMLDynamics::update";

				// ---------------------------------------------------------------------------------------

				// first-time-through initialization
				if( getTimeStep() == 0 ) {
					std::stringstream message;
					message << methodName;
					int errors = 0;

					// invert masses
					for( int ii = 0; ii < numberOfAtoms; ii++ ) {
						if( masses[ii] <= 0.0 ) {
							message << "mass at atom index=" << ii << " (" << masses[ii] << ") is <= 0" << std::endl;
							errors++;
						} else {
							inverseMasses[ii] = 1.0 / masses[ii];
						}
					}

					// exit if errors
					if( errors ) {
						SimTKOpenMMLog::printError( message );
					}
				}

				switch( stepType ) {
					case 1: {
						// Update the velocity.
						double deltaT = getDeltaT();
						double tau = mTau;
						const double vscale = EXP( -deltaT / tau );
						const double fscale = ( 1 - vscale ) * tau;
						const double noisescale = std::sqrt( BOLTZ * getTemperature() * ( 1 - vscale * vscale ) );
						for( int i = 0; i < numberOfAtoms; i++ )
							for( int j = 0; j < 3; j++ )
								velocities[i][j] = vscale * velocities[i][j] + fscale * forces[i][j] * inverseMasses[i] +
												   noisescale * SimTKOpenMMUtilities::getNormallyDistributedRandomNumber() * std::sqrt( inverseMasses[i] );

						// Project resulting velocities onto subspace
						subspaceProjection( velocities, velocities, masses, inverseMasses, false );

						// Update the positions.
						for( int i = 0; i < numberOfAtoms; i++ )
							for( int j = 0; j < 3; j++ ) {
								atomCoordinates[i][j] += deltaT * velocities[i][j];
							}
						break;
					}
					case 2:
						// Do nothing.
						break;

						//simple minimizer step, assume quadratic line search value is correct
						//accept move if new PE < old PE
					case 3: {
						//save current PE in case quadratic required
						lastPE = currentPE;

						//project forces into complement space, put in xPrime
						subspaceProjection( forces, xPrime, inverseMasses, masses, true );
						if( minimizerScale != 1.0 )
							for( int ii = 0; ii < numberOfAtoms; ii++ )
								for( int jj = 0; jj < 3; jj++ ) {
									xPrime[ii][jj] *= minimizerScale;
								}

						//Add minimizer position update to atomCoordinates
						// with 'line search guess = 1/maxEig (the solution if the system was quadratic)
						for( int ii = 0; ii < numberOfAtoms; ii++ ) {
							double factor = inverseMasses[ii] / _maxEig;

							atomCoordinates[ii][0] += factor * xPrime[ii][0];
							atomCoordinates[ii][1] += factor * xPrime[ii][1];
							atomCoordinates[ii][2] += factor * xPrime[ii][2];
						}
						break;
					}

					//quadratic correction if simple minimizer new PE value is greater than the old PE value
					case 4: {
						//We assume the move direction is still in xPrime

						//Get quadratic 'line search' value
						double lambda = ( double )( 1.0 / _maxEig );
						double oldLambda = lambda;

						//Solve quadratic for slope at new point
						//get slope dPE/d\lambda for quadratic, just equal to minus dot product of 'proposed position move' and forces (=-\nabla PE)
						double newSlope = 0.0;
						for( int ii = 0; ii < numberOfAtoms; ii++ ) {
							for( int jj = 0; jj < 3; jj++ ) {
								newSlope -= xPrime[ii][jj] * forces[ii][jj] * inverseMasses[ii];
							}
						}

						//solve for minimum for quadratic fit using two PE vales and the slope with /lambda=0
						//for 'newSlope' use PE=a(\lambda_e-\lambda)^2+b(\lambda_e-\lambda)+c, \lambda_e is 1/maxEig.
						double a, b;

						//a = (((currentPE - lastPE) / lambda - lastSlope) / lambda);
						a = ( ( ( lastPE - currentPE ) / oldLambda + newSlope ) / oldLambda );
						//b = lastSlope;
						b = -newSlope;

						//calculate \lambda at minimum of quadratic fit
						if( a != 0.0 ) {
							//lambda = -b / (2 * a);
							lambda = b / ( 2 * a ) + oldLambda;
						} else {
							lambda = ( double )( oldLambda / 2.0 );
						}

						//test if lambda negative, if so just use smaller lambda
						if( lambda <= 0.0 ) {
							lambda = ( double )( oldLambda / 2.0 );
						}

						//Remove previous position update (-oldLambda) and add new move (lambda)
						for( int ii = 0; ii < numberOfAtoms; ii++ ) {
							const double factor = inverseMasses[ii] * ( lambda - oldLambda );

							atomCoordinates[ii][0] += factor * xPrime[ii][0];
							atomCoordinates[ii][1] += factor * xPrime[ii][1];
							atomCoordinates[ii][2] += factor * xPrime[ii][2];
						}
						break;
					}
					case 5: {
						RejectStep( atomCoordinates );
						break;
					}
					case 6: {
						AcceptStep( atomCoordinates );
						break;
					}
				}
				incrementTimeStep();

				return 0;

			}

			void Dynamics::Integrate() {
				
			}
			
			void Dynamics::UpdateTime() {
				
			}
			
			void Dynamics::AcceptStep( VectorArray& coordinates ) {
				const unsigned int atoms = oldPositions.size();
				for( unsigned int i = 0; i < atoms; i++ ) {
					oldPositions[i][0] = coordinates[i][0];
					oldPositions[i][1] = coordinates[i][1];
					oldPositions[i][2] = coordinates[i][2];
				}
				minimizerScale = 1.0;
			}
			
			void Dynamics::RejectStep( VectorArray& coordinates ) {
				const unsigned int atoms = coordinates.size();
				for( int i = 0; i < atoms; i++ ) {
					coordinates[i][0] = oldPositions[i][0];
					coordinates[i][1] = oldPositions[i][1];
					coordinates[i][2] = oldPositions[i][2];
				}
				minimizerScale *= 0.25;
			}
			
			void Dynamics::LinearMinimize() {
				
			}
			
			void Dynamics::QuadraticMinimize() {
				
			}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Find forces OR positions inside subspace (defined as the span of the 'eigenvectors' Q)
// Take 'array' as input, 'outArray' as output (may be the same vector).
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			void Dynamics::subspaceProjection( VectorArray& array,
											   VectorArray& outArray,
											   DoubleArray& scale,
											   DoubleArray& inverseScale,
											   bool projectIntoComplement ) {

				const unsigned int AtomCount = array.size();
	
				//If 'array' and 'outArray are not the same array
				//copy 'array' into outArray
				const unsigned int _3N = AtomCount * 3;
				if( &array != &outArray ) {
					for( unsigned int i = 0; i < AtomCount; i++ ) {
						for( unsigned int j = 0; j < 3; j++ ) {
							outArray[i][j] = array[i][j];
						}
					}
				}

				//~~~~We need to calculate M^{1/2}QQ^TM^{-1/2}force or M^{-1/2}QQ^TM^{1/2}positions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

				//First we weight the array by multiplying by
				//the square-root of the atomic masses for positions
				//or the the inverse of the square-root of the atomic masses for forces.
				//
				//a'=M^{-1/2}*f for forces, OR a'= M^{1/2}*x for positions
				//
				for( int i = 0; i < AtomCount; i++ ) {        //N loops
					double  weight = std::sqrt( scale[i] );
					for( unsigned int j = 0; j < 3; j++ ) {         //times 3 loops
						outArray[i][j] *= weight;
					}
				}

				//Project onto mode space by taking the matrix product of
				//the transpose of the eigenvectors Q with the array.
				//
				//c=Q^T*a', a' from last algorithm step
				//

				//If no Blas is available we need to manually find the product c=A*b
				//c_i=\sum_{j=1}^n A_{i,j} b_j

				//c=Q^T*a', a' from last algorithm step
				//Q is a linear array in column major format
				//so tmpC_i = \sum_{j=1}^n Q_{j,i} outArray_j
				//Q_{j,i}=_projectionVectors[j * numberOfAtoms * 3 + i]

				DoubleArray tmpC( _numProjectionVectors );
				for( int i = 0; i < ( int ) _numProjectionVectors; i++ ) { //over all eigenvectors

					tmpC[i] = 0.0;  //clear

					for( int j = 0; j < ( int ) _3N; j++ ) { //over each element in the vector
						tmpC[i] += _projectionVectors[j  + i * _3N] * outArray[j / 3][j % 3];
					}
				}

				//Now find projected force/positions a'' by matrix product with Eigenvectors Q
				//a''=Qc
				//so outArray_i  = \sum_{j=1}^n Q_{i,j} tmpC_i

				//find product
				for( int i = 0; i < ( int ) _3N; i++ ) { //over each element in the vector

					//if sub-space do Q*c
					//else do a'-Q(Q^T a') = (I-QQ^T)a'
					const int ii = i / 3;
					const int jj = i % 3;
					if( !projectIntoComplement ) {
						outArray[ii][jj] = 0.0; //if not complement

						for( int j = 0; j < _numProjectionVectors; j++ ) { //over all eigenvectors
							outArray[ii][jj] += _projectionVectors[i + j * _3N] * tmpC[j];
						}
					} else {
						for( int j = 0; j < _numProjectionVectors; j++ ) { //over all eigenvectors
							outArray[ii][jj] -= _projectionVectors[i + j * _3N] * tmpC[j];
						}

					}

				}

				//Finally we weight the array by multiplying by
				//the inverse of the square-root of the atomic masses for positions
				//or the the square-root of the atomic masses for forces.
				//
				//a'''=M^{1/2}*a'' or a'''=M^{-1/2}*a''
				//
				for( int i = 0; i < AtomCount; i++ ) {
					double  unweight = std::sqrt( inverseScale[i] );
					for( unsigned int j = 0; j < 3; j++ ) {
						outArray[i][j] *= unweight;
					}
				}
			}
		}
	}
}
