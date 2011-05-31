#ifndef __ReferenceNMLDynamics_H__
#define __ReferenceNMLDynamics_H__

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

namespace OpenMM_LTMD {

class ReferenceNMLDynamics : public ReferenceDynamics {

   private:

      std::vector<OpenMM::RealVec> xPrime;
      std::vector<OpenMM::RealVec> oldPositions;
      std::vector<RealOpenMM> inverseMasses;

      RealOpenMM _tau;
      RealOpenMM* _projectionVectors;
      unsigned int _numProjectionVectors;
      RealOpenMM _minimumLimit, _maxEig;
      RealOpenMM lastPE, lastSlope, minimizerScale;
      
   public:

      /**---------------------------------------------------------------------------------------
      
         Constructor

         @param numberOfAtoms  number of atoms
         @param deltaT         delta t for dynamics
         @param tau            viscosity
         @param temperature    temperature
      
         --------------------------------------------------------------------------------------- */

       ReferenceNMLDynamics( int numberOfAtoms, RealOpenMM deltaT, RealOpenMM tau, RealOpenMM temperature,
                   RealOpenMM* projectionVectors, unsigned int numProjectionVectors, 
                   RealOpenMM minimumLimit, RealOpenMM maxEig
                    );

      /**---------------------------------------------------------------------------------------
      
         Destructor
      
         --------------------------------------------------------------------------------------- */

       ~ReferenceNMLDynamics( );

      /**---------------------------------------------------------------------------------------
      
         Get tau
      
         @return tau
      
         --------------------------------------------------------------------------------------- */
      
      RealOpenMM getTau( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Update
      
         @param numberOfAtoms       number of atoms
         @param atomCoordinates     atom coordinates
         @param velocities          velocities
         @param forces              forces
         @param masses              atom masses
      
         @return ReferenceDynamics::DefaultReturn
      
         --------------------------------------------------------------------------------------- */
     
      int update( int numberOfAtoms, std::vector<OpenMM::RealVec>& atomCoordinates,
                  std::vector<OpenMM::RealVec>& velocities, std::vector<OpenMM::RealVec>& forces,
                  std::vector<RealOpenMM>& masses, const RealOpenMM currentPE, const int stepType );

      /**---------------------------------------------------------------------------------------
       Find forces OR positions inside subspace (defined as the span of the 'eigenvectors' Q)
       Take 'array' as input, 'outArray' as output (may be the same vector).
       ----------------------------------------------------------------------------------------- */
      void subspaceProjection(  std::vector<OpenMM::RealVec>& arrayParam,
                                std::vector<OpenMM::RealVec>& outArrayParam,
                                int numberOfAtoms,
                                std::vector<RealOpenMM>& scale,
                                std::vector<RealOpenMM>& inverseScale,
                                bool projectIntoComplement);
  
};

// ---------------------------------------------------------------------------------------

} // namespace OpenMM_LTMD

#endif // __ReferenceNMLDynamics_H__
