#ifndef OPENMM_LTMD_REFERENCE_DYNAMICS_H_
#define OPENMM_LTMD_REFERENCE_DYNAMICS_H_

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

#include "SimTKUtilities/SimTKOpenMMRealType.h"
#include "SimTKReference/ReferenceDynamics.h"

// ---------------------------------------------------------------------------------------

namespace OpenMM {
	namespace LTMD {
		namespace Reference {
			typedef std::vector<double> DoubleArray;
			typedef std::vector<OpenMM::RealVec> VectorArray;
			
			class Dynamics : public ReferenceDynamics {
				private:
					const double mTau;
					const unsigned int mAtomCount;
					
					VectorArray xPrime, oldPositions;
					DoubleArray inverseMasses;

					double *_projectionVectors;
					unsigned int _numProjectionVectors;
					double _minimumLimit, _maxEig;
					double lastPE, lastSlope, minimizerScale;
				public:
					/**---------------------------------------------------------------------------------------

					   Constructor

					   @param numberOfAtoms  number of atoms
					   @param deltaT         delta t for dynamics
					   @param tau            viscosity
					   @param temperature    temperature

					   --------------------------------------------------------------------------------------- */

					Dynamics( int numberOfAtoms, double deltaT, double tau, double temperature,
							  double *projectionVectors, unsigned int numProjectionVectors,
							  double maxEig
							);

					~Dynamics( );

					void SetMaxEigenValue( double value );

					/**---------------------------------------------------------------------------------------

					   Update

					   @param numberOfAtoms       number of atoms
					   @param atomCoordinates     atom coordinates
					   @param velocities          velocities
					   @param forces              forces
					   @param masses              atom masses

					   @return ReferenceDynamics::DefaultReturn

					   --------------------------------------------------------------------------------------- */
					
					int update( VectorArray& atomCoordinates, VectorArray& velocities, VectorArray& forces, DoubleArray& masses, const double currentPE, const int stepType );

				private:
					/**---------------------------------------------------------------------------------------
					 Find forces OR positions inside subspace (defined as the span of the 'eigenvectors' Q)
					 Take 'array' as input, 'outArray' as output (may be the same vector).
					 ----------------------------------------------------------------------------------------- */
					void Project( const VectorArray& in, VectorArray& out, const DoubleArray& scale, const DoubleArray& inverseScale, const bool compliment );

					void Integrate( VectorArray& coordinates, VectorArray& velocities, const VectorArray& forces, const DoubleArray& masses );
					void UpdateTime();

					void AcceptStep( VectorArray& coordinates );
					void RejectStep( VectorArray& coordinates );

					void LinearMinimize( const double energy, VectorArray& coordinates, const VectorArray& forces, const DoubleArray& masses );
					void QuadraticMinimize( const double energy, VectorArray& coordinates, const VectorArray& forces );
			};
		}
	}
}

#endif // OPENMM_LTMD_REFERENCE_DYNAMICS_H_
