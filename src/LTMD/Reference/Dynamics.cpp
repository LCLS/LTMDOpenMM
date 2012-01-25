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
						Integrate( atomCoordinates, velocities, forces, masses );
						break;
					}
					case 2:{
						UpdateTime();
						break;
					}
					case 3: {
						LinearMinimize( currentPE, atomCoordinates, forces, masses );
						break;
					}
					case 4: {
						QuadraticMinimize( currentPE, atomCoordinates, forces );
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

			void Dynamics::Integrate( VectorArray& coordinates, VectorArray& velocities, const VectorArray& forces, const DoubleArray& masses ) {
				// Calculate Constants
				const double deltaT = getDeltaT();
				const double vscale = EXP( -deltaT / mTau );
				const double fscale = ( 1 - vscale ) * mTau;
				const double noisescale = std::sqrt( BOLTZ * getTemperature() * ( 1 - vscale * vscale ) );
				
				// Update the velocity.
				const unsigned int vatoms = velocities.size();
				for( unsigned int i = 0; i < vatoms; i++ ){
					for( unsigned int j = 0; j < 3; j++ ) {
						const double velocity = vscale * velocities[i][j];
						const double force = fscale * forces[i][j];
						const double noise = noisescale * SimTKOpenMMUtilities::getNormallyDistributedRandomNumber();
						
						velocities[i][j] = velocity + force * inverseMasses[i] + noise * std::sqrt( inverseMasses[i] );
					}
				}
				
				// Project resulting velocities onto subspace
				subspaceProjection( velocities, velocities, masses, inverseMasses, false );
				
				// Update the positions.
				const unsigned int atoms = coordinates.size();
				for( unsigned int i = 0; i < atoms; i++ ){
					coordinates[i][0] += deltaT * velocities[i][0];
					coordinates[i][1] += deltaT * velocities[i][1];
					coordinates[i][2] += deltaT * velocities[i][2];
				}
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
			
			void Dynamics::LinearMinimize( const double energy, VectorArray& coordinates, const VectorArray& forces, const DoubleArray& masses ) {
				//save current PE in case quadratic required
				lastPE = energy;
				
				//project forces into complement space, put in xPrime
				subspaceProjection( forces, xPrime, inverseMasses, masses, true );
				
				// Scale xPrime if needed
				if( minimizerScale != 1.0 ){
					const unsigned int atoms = xPrime.size();
					for( unsigned int i = 0; i < atoms; i++ ){
						xPrime[i][0] *= minimizerScale;
						xPrime[i][1] *= minimizerScale;
						xPrime[i][2] *= minimizerScale;
					}
				}
				
				//Add minimizer position update to atomCoordinates
				// with 'line search guess = 1/maxEig (the solution if the system was quadratic)
				const unsigned int atoms = coordinates.size();
				for( unsigned int i = 0; i < atoms; i++ ) {
					double factor = inverseMasses[i] / _maxEig;
					
					coordinates[i][0] += factor * xPrime[i][0];
					coordinates[i][1] += factor * xPrime[i][1];
					coordinates[i][2] += factor * xPrime[i][2];
				}
			}
			
			void Dynamics::QuadraticMinimize( const double energy, VectorArray& coordinates, const VectorArray& forces ) {
				//Get quadratic 'line search' value
				double lambda = 1.0 / _maxEig;
				const double oldLambda = lambda;
				
				//Solve quadratic for slope at new point
				//get slope dPE/d\lambda for quadratic, just equal to minus dot product of 'proposed position move' and forces (=-\nabla PE)
				double newSlope = 0.0;
				
				const unsigned int xatoms = xPrime.size();
				for( unsigned int i = 0; i < xatoms; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						newSlope -= xPrime[i][j] * forces[i][j] * inverseMasses[i];
					}
				}
				
				//solve for minimum for quadratic fit using two PE vales and the slope with /lambda=0
				//for 'newSlope' use PE=a(\lambda_e-\lambda)^2+b(\lambda_e-\lambda)+c, \lambda_e is 1/maxEig.
				const double a = ( ( ( lastPE - energy ) / oldLambda + newSlope ) / oldLambda );
				const double b = -newSlope;
				
				//calculate \lambda at minimum of quadratic fit
				if( a != 0.0 ) {
					lambda = b / ( 2 * a ) + oldLambda;
				} else {
					lambda = oldLambda / 2.0;
				}
				
				//test if lambda negative, if so just use smaller lambda
				if( lambda <= 0.0 ) {
					lambda = oldLambda / 2.0;
				}
				
				//Remove previous position update (-oldLambda) and add new move (lambda)
				const unsigned int atoms = coordinates.size();
				for( unsigned int i = 0; i < atoms; i++ ) {
					const double factor = inverseMasses[i] * ( lambda - oldLambda );
					
					coordinates[i][0] += factor * xPrime[i][0];
					coordinates[i][1] += factor * xPrime[i][1];
					coordinates[i][2] += factor * xPrime[i][2];
				}
			}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Find forces OR positions inside subspace (defined as the span of the 'eigenvectors' Q)
// Take 'array' as input, 'outArray' as output (may be the same vector).
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			void Dynamics::subspaceProjection( const VectorArray& in,
											   VectorArray& out,
											   const DoubleArray& scale,
											   const DoubleArray& inverseScale,
											   const bool projectIntoComplement ) {

				const unsigned int AtomCount = in.size();
	
				//If 'array' and 'outArray are not the same array
				//copy 'array' into outArray
				const unsigned int _3N = AtomCount * 3;
				if( &in != &out ) {
					for( unsigned int i = 0; i < AtomCount; i++ ) {
						for( unsigned int j = 0; j < 3; j++ ) {
							out[i][j] = in[i][j];
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
						out[i][j] *= weight;
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
						tmpC[i] += _projectionVectors[j  + i * _3N] * out[j / 3][j % 3];
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
						out[ii][jj] = 0.0; //if not complement

						for( int j = 0; j < _numProjectionVectors; j++ ) { //over all eigenvectors
							out[ii][jj] += _projectionVectors[i + j * _3N] * tmpC[j];
						}
					} else {
						for( int j = 0; j < _numProjectionVectors; j++ ) { //over all eigenvectors
							out[ii][jj] -= _projectionVectors[i + j * _3N] * tmpC[j];
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
						out[i][j] *= unweight;
					}
				}
			}
		}
	}
}
