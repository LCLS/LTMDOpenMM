/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009 Stanford University and the Authors.           *
 * Authors: Scott Le Grand, Peter Eastman                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "CudaIntegrationUtilities.h"
#include "CudaContext.h"
#include "CudaArray.h"
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <vector_functions.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <stdlib.h>
using namespace std;
using namespace OpenMM;


// CPU code
void kNMLUpdate( CUmodule *module, CudaContext *cu, float deltaT, float tau, float kT, int numModes, int &iterations, CudaArray &modes, CudaArray &modeWeights, CudaArray &noiseVal ) {
	int atoms = cu->getNumAtoms();
	int paddednumatoms = cu->getPaddedNumAtoms();
	int randomIndex = cu->getIntegrationUtilities().prepareRandomNumbers(cu->getPaddedNumAtoms());

	CUfunction update1Kernel = cu->getKernel( *module, "kNMLUpdate1_kernel" );
	void *update1Args[] = {
		&atoms, &paddednumatoms, &tau, &deltaT, &kT,
		&cu->getPosq().getDevicePointer(), &noiseVal.getDevicePointer(), &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &cu->getIntegrationUtilities().getRandom().getDevicePointer(), &randomIndex
	};
	cu->executeKernel( update1Kernel, update1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );

	CUfunction update2Kernel = cu->getKernel( *module, "kNMLUpdate2_kernel" );
	void *update2Args[] = {&atoms, &numModes, &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
	cu->executeKernel( update2Kernel, update2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize * sizeof( float ) );

	CUfunction update3Kernel = cu->getKernel( *module, "kNMLUpdate3_kernel" );
	void *update3Args[] = {&atoms, &numModes, &deltaT, &cu->getPosq().getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer(), &noiseVal.getDevicePointer()};
	cu->executeKernel( update3Kernel, update3Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes * sizeof( float ) );

}

#ifdef FAST_NOISE
void kFastNoise( CUmodule *module, CudaContext *cu, int numModes, float kT, int &iterations, CudaArray &modes, CudaArray &modeWeights, float maxEigenvalue, CudaArray &noiseVal, CudaArray &oldpos, float stepSize ) {
	int atoms = cu->getNumAtoms();
	int paddednumatoms = cu->getPaddedNumAtoms();
	int randomIndex = cu->getIntegrationUtilities().prepareRandomNumbers(cu->getPaddedNumAtoms());

	CUfunction fastnoise1Kernel = cu->getKernel( *module, "kFastNoise1_kernel" );
	void *fastnoise1Args[] = {
		&atoms, &paddednumatoms, &numModes, &kT, &oldpos.getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer(), &cu->getIntegrationUtilities().getRandom().getDevicePointer(), &randomIndex, &maxEigenvalue, &stepSize
	};
	cu->executeKernel( fastnoise1Kernel, fastnoise1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize * sizeof( float ) );

	CUfunction fastnoise2Kernel = cu->getKernel( *module, "kFastNoise2_kernel" );
	void *fastnoise2Args[] = {
		&atoms, &numModes, &cu->getPosq().getDevicePointer(), &noiseVal.getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()
	};
	cu->executeKernel( fastnoise2Kernel, fastnoise2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes * sizeof( float ) );
}
#endif

void kNMLRejectMinimizationStep( CUmodule *module, CudaContext *cu, CudaArray &oldpos ) {
	int atoms = cu->getNumAtoms();

	CUfunction rejectKernel = cu->getKernel( *module, "kRejectMinimizationStep_kernel" );
	void *rejectArgs[] = {&atoms, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer() };
	cu->executeKernel( rejectKernel, rejectArgs, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
}

void kNMLAcceptMinimizationStep( CUmodule *module, CudaContext *cu, CudaArray &oldpos ) {
	int atoms = cu->getNumAtoms();

	CUfunction acceptKernel = cu->getKernel( *module, "kAcceptMinimizationStep_kernel" );
	void *acceptArgs[] = {&atoms, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer() };
	cu->executeKernel( acceptKernel, acceptArgs, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
}

void kNMLLinearMinimize( CUmodule *module, CudaContext *cu, int numModes, float maxEigenvalue, CudaArray &oldpos, CudaArray &modes, CudaArray &modeWeights ) {
	int atoms = cu->getNumAtoms();
	int paddedatoms = cu->getPaddedNumAtoms();
	float oneoverEig = 1.0f / maxEigenvalue;

	CUfunction linmin1Kernel = cu->getKernel( *module, "kNMLLinearMinimize1_kernel" );
	void *linmin1Args[] = {&atoms, &paddedatoms, &numModes, &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
	cu->executeKernel( linmin1Kernel, linmin1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize * sizeof( float ) );

	CUfunction linmin2Kernel = cu->getKernel( *module, "kNMLLinearMinimize2_kernel" );
	void *linmin2Args[] = {&atoms, &paddedatoms, &numModes, &oneoverEig, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer(), &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
	cu->executeKernel( linmin2Kernel, linmin2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes * sizeof( float ) );
}

void kNMLQuadraticMinimize( CUmodule *module, CudaContext *cu, float maxEigenvalue, float currentPE, float lastPE, CudaArray &oldpos, CudaArray &slopeBuffer, CudaArray &lambdaval ) {
	int atoms = cu->getNumAtoms();
	int paddedatoms = cu->getPaddedNumAtoms();
	float oneoverEig = 1.0f / maxEigenvalue;

	CUfunction quadmin1Kernel = cu->getKernel( *module, "kNMLQuadraticMinimize1_kernel" );
	void *quadmin1Args[] = {&atoms, &paddedatoms, &oldpos.getDevicePointer(),&cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(),&slopeBuffer.getDevicePointer()};
	cu->executeKernel( quadmin1Kernel, quadmin1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize * sizeof( float ) );

	CUfunction quadmin2Kernel = cu->getKernel( *module, "kNMLQuadraticMinimize2_kernel" );
	void *quadmin2Args[] = {&atoms, &currentPE, &lastPE, &oneoverEig, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer(), &cu->getVelm().getDevicePointer(), &slopeBuffer.getDevicePointer(), &lambdaval.getDevicePointer()};
	cu->executeKernel( quadmin2Kernel, quadmin2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->getNumThreadBlocks()*cu->ThreadBlockSize * sizeof( float ) );
}
