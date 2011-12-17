#ifndef OPENMM_LTMD_ANALYSIS_H_
#define OPENMM_LTMD_ANALYSIS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
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

#include <map>
#include <utility>
#include <vector>

#include "OpenMM.h"
#include "jama_eig.h"
#include "LTMD/Parameters.h"

namespace OpenMM {
	namespace LTMD {
		class OPENMM_EXPORT Analysis {
			public:
				Analysis() : mParticleCount( 0 ), mLargestBlockSize( -1 ) {
					mInitialized = false;
					blockContext = NULL;
				}
				~Analysis() {
					if( blockContext ) {
						delete blockContext;
					}
				}
				void computeEigenvectorsFull( ContextImpl &contextImpl, Parameters *ltmd );
				const std::vector<std::vector<Vec3> >& getEigenvectors() const {
					return eigenvectors;
				}
				double getMaxEigenvalue() const {
					return maxEigenvalue;
				}
				unsigned int blockNumber( int );
				bool inSameBlock( int, int, int, int );
			private:
				void Initialize( Context &context, const Parameters &ltmd );
				void DiagonalizeBlock( const unsigned int block, const TNT::Array2D<double>& hessian, 
					const std::vector<Vec3>& positions, TNT::Array1D<double>& eval, TNT::Array2D<double>& evec );
			private:
				static double getDelta( double value, bool isDoublePrecision, Parameters *ltmd );

				unsigned int mParticleCount;
				std::vector<double> mParticleMass;
				
				int mLargestBlockSize;
				bool mInitialized;
				std::vector<std::pair<int, int> > bonds;
				std::vector<std::vector<int> > particleBonds;
				std::vector<std::vector<double> > projection;
				std::vector<std::vector<Vec3> > eigenvectors;
				double maxEigenvalue;
				Context *blockContext;
				std::vector<int> blocks;
		};
	}
}

#endif // OPENMM_LTMD_ANALYSIS_H_
