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

//#include "kernels/gputypes.h"
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
#include <random>
using namespace std;
using namespace OpenMM;
//#define LAUNCHERROR(s) \
//    { \
//        cudaError_t status = cudaGetLastError(); \
//        if (status != cudaSuccess) { \
//            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
//            exit(-1); \
//        } \
//    }

/*typedef float Real;

__global__ void kNMLUpdate1_kernel( int numAtoms, int paddedNumAtoms, float tau, float dt, float kT, float* posq, float* posqP, float4* velm, float4* force,
									float4 *random, int* randomPosition, int totalRandoms ) {
	// Update the velocity.
	const Real vscale = exp( -dt / tau );
	const Real fscale = ( 1.0f - vscale ) * tau;
	const Real noisescale = sqrt( kT * ( 1 - vscale * vscale ) );

	int rpos = randomPosition[blockIdx.x];
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const float4 n = random[rpos + atom];
		const float4 randomNoise = make_float4( n.x * noisescale, n.y * noisescale, n.z * noisescale, n.w * noisescale );

		const Real sqrtInvMass = sqrt( velm[atom].w );

		float4 v = velm[atom];
		float4 f = force[atom];

		v.x = ( vscale * v.x ) + ( fscale * f.x * v.w ) + ( randomNoise.x * sqrtInvMass );
		v.y = ( vscale * v.y ) + ( fscale * f.y * v.w ) + ( randomNoise.y * sqrtInvMass );
		v.z = ( vscale * v.z ) + ( fscale * f.z * v.w ) + ( randomNoise.z * sqrtInvMass );

		velm[atom] = v;
	}

	if( threadIdx.x == 0 ) {
		rpos += paddedNumAtoms;
		if( rpos > totalRandoms ) {
			rpos -= totalRandoms;
		}
		randomPosition[blockIdx.x] = rpos;
	}
}

__global__ void kNMLUpdate2_kernel( int numAtoms, int numModes, float4 *velm, float4 *modes, float *modeWeights ) {
	extern __shared__ float dotBuffer[];
	for( int mode = blockIdx.x; mode < numModes; mode += gridDim.x ) {
		// Compute the projection of the mass weighted velocity onto one normal mode vector.
		Real dot = 0.0f;

		for( int atom = threadIdx.x; atom < numAtoms; atom += blockDim.x ) {
			const int modePos = mode * numAtoms + atom;
			const Real scale = 1.0f / sqrt( velm[atom].w );

			float4 v = velm[atom];
			float4 m = modes[modePos];

			dot += scale * ( v.x * m.x + v.y * m.y + v.z * m.z );
		}

		dotBuffer[threadIdx.x] = dot;

		__syncthreads();
		if( threadIdx.x == 0 ) {
			Real sum = 0;
			for( int i = 0; i < blockDim.x; i++ ) {
				sum += dotBuffer[i];
			}
			modeWeights[mode] = sum;
		}
	}
}

__global__ void kNMLUpdate3_kernel( int numAtoms, int numModes, float dt, float4 *posq, float4 *velm, float4 *modes, float *modeWeights, float4 *noiseVal ) {
	// Load the weights into shared memory.
	extern __shared__ float weightBuffer[];
	for( int mode = threadIdx.x; mode < numModes; mode += blockDim.x ) {
		weightBuffer[mode] = modeWeights[mode];
	}
	__syncthreads();

	// Compute the projected velocities and update the atom positions.
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real invMass = velm[atom].w, scale = sqrt( invMass );

		float3 v = make_float3( 0.0f, 0.0f, 0.0f );
		for( int mode = 0; mode < numModes; mode++ ) {
			float4 m = modes[mode * numAtoms + atom];
			float weight = weightBuffer[mode];
			v.x += m.x * weight;
			v.y += m.y * weight;
			v.z += m.z * weight;
		}

		v.x *= scale;
		v.y *= scale;
		v.z *= scale;
		velm[atom] = make_float4( v.x, v.y, v.z, invMass );

		float4 pos = posq[atom];

		// Add Step
		pos.x += dt * v.x;
		pos.y += dt * v.y;
		pos.z += dt * v.z;

#ifdef FAST_NOISE
		// Remove Noise
		pos.x -= noiseVal[atom].x;
		pos.y -= noiseVal[atom].y;
		pos.z -= noiseVal[atom].z;
#endif

		posq[atom] = pos;
	}
}
*/

// TMC the helper function is actually gone, just the kernel is left
//extern void kGenerateRandoms( gpuContext gpu );


/*void kNMLUpdate( gpuContext gpu, int numModes, CUDAStream<float4>& modes, CUDAStream<float>& modeWeights, CUDAStream<float4>& noiseVal ) {
	kNMLUpdate1_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block >>> ( gpu->natoms, gpu->sim.paddedNumberOfAtoms,
		    gpu->sim.tau, gpu->sim.deltaT, gpu->sim.kT, gpu->sim.pPosq, noiseVal._pDevData, gpu->sim.pVelm4, gpu->sim.pForce4, gpu->sim.pRandom4, gpu->sim.pRandomPosition, gpu->sim.randoms );
	LAUNCHERROR( "kNMLUpdate1" );
	kNMLUpdate2_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, gpu->sim.update_threads_per_block *sizeof( float ) >>> ( gpu->natoms,
			numModes, gpu->sim.pVelm4, modes._pDevData, modeWeights._pDevData );
	LAUNCHERROR( "kNMLUpdate2" );
	kNMLUpdate3_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, numModes *sizeof( float ) >>> ( gpu->natoms, numModes,
			gpu->sim.deltaT, gpu->sim.pPosq, gpu->sim.pVelm4, modes._pDevData, modeWeights._pDevData, noiseVal._pDevData );
	LAUNCHERROR( "kNMLUpdate3" );

	// Update randoms if necessary
	gpu->iterations++;
	if( gpu->iterations == gpu->sim.randomIterations ) {
		kGenerateRandoms( gpu );
		gpu->iterations = 0;
	}
}*/


// CPU code
void kNMLUpdate(CUmodule* module, CudaContext* cu, float deltaT, float tau, float kT, int numModes, int& iterations, CudaArray& modes, CudaArray& modeWeights, CudaArray& noiseVal, CudaArray& randomIndex ) {
	// TMC not sure at the moment about random data, and cannot find kGEnerateRandom()
	// Still need to figure that part ou	
	// gpu->sim.pRandom4
        // gpu->sim.pRandomPosition
        // gpu->sim.random = cu.getIntegrationUtilities().getRandom().getDevicePointer()
	int atoms = cu->getNumAtoms();
	int paddednumatoms = cu->getPaddedNumAtoms();
	printf("Calling NMLUpdate with these values: %f %f %f\n", deltaT, tau, kT);
	void* update1Args[] = {&atoms, &paddednumatoms, &tau, &deltaT, &kT, 
        //CUmodule module = cu->createModule(CudaLTMDKernelSources::NMLupdates);
                 //           &cu->getPosq().getDevicePointer(), &noiseVal.getDevicePointer(), &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &cu->getIntegrationUtilities().getRandom().getDevicePointer(), &randomPos, &atoms }; // # of random numbers equal to the number of atoms? TMC
                            &cu->getPosq().getDevicePointer(), &noiseVal.getDevicePointer(), &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &cu->getIntegrationUtilities().getRandom().getDevicePointer(), &randomIndex.getDevicePointer(), &atoms }; // # of random numbers equal to the number of atoms? TMC
	CUfunction update1Kernel, update2Kernel, update3Kernel;
        update1Kernel = cu->getKernel(*module, "kNMLUpdate1_kernel");
	// TMC nan gets generated when this kernel is included...check
	int i = 0;
	//float4* v = new float4[paddednumatoms];
	//cu->getForce().download(v);
	//for (i = 0; i < atoms; i++) {
        //    printf("FORCE BEFORE %d: %f %f %f %f\n", i, v[i].w, v[i].x, v[i].y, v[i].z);
        //}
	cu->executeKernel(update1Kernel, update1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize);
	// TMC Velocities are way off after this
        //cu->getForce().download(v);
	/*for (i = 0; i < atoms; i++) {
            printf("VEL AFTER %d: %f %f %f %f\n", i, v[i].w, v[i].x, v[i].y, v[i].z);
        }*/
 
	//LAUNCHERROR( "kNMLUpdate1" );


        void* update2Args[] = {&atoms, &numModes, &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
	update2Kernel = cu->getKernel(*module, "kNMLUpdate2_kernel");
	cu->executeKernel(update2Kernel, update2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(float)); 
	//LAUNCHERROR( "kNMLUpdate2" );
	
        void* update3Args[] = {&atoms, &numModes, &deltaT, &cu->getPosq().getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer(), &noiseVal.getDevicePointer()};
	update3Kernel = cu->getKernel(*module, "kNMLUpdate3_kernel");
	cu->executeKernel(update3Kernel, update3Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes*sizeof(float)); 
	//LAUNCHERROR( "kNMLUpdate3" );

	// Update randoms if necessary
	// Again, assuming 20 for randomIterations because it was set to that in the construct and never changed by our integrator
	//int randomIterations = 20;
	//iterations++;
	//if( iterations == randomIterations ) {
		//kGenerateRandoms( gpu );
	//	cu->getIntegrationUtilities().prepareRandomNumbers( paddednumatoms   ); // 
	//	iterations = 0;
	//}
}

#ifdef FAST_NOISE
/*__global__ void kFastNoise1_kernel( int numAtoms, int paddedNumAtoms, int numModes, float kT, float4 *noiseVal, float4 *velm, float4 *modes, float *modeWeights, float4 *random, int *randomPosition, int totalRandoms, float maxEigenvalue, float stepSize ) {
	extern __shared__ float dotBuffer[];
	const Real val = stepSize / 0.002;
	const Real noisescale = sqrt( 2 * kT * 1.0f / maxEigenvalue );

	int rpos = randomPosition[blockIdx.x];
	for( int mode = blockIdx.x; mode < numModes; mode += gridDim.x ) {
		Real dot = 0.0f;

		for( int atom = threadIdx.x; atom < numAtoms; atom += blockDim.x ) {
			const float4 n = random[rpos + atom];
			const float4 randomNoise = make_float4( n.x * noisescale, n.y * noisescale, n.z * noisescale, n.w * noisescale );

			noiseVal[atom] = randomNoise;

			float4 m = modes[mode * numAtoms + atom];
			dot += randomNoise.x * m.x + randomNoise.y * m.y + randomNoise.z * m.z;
		}

		dotBuffer[threadIdx.x] = dot;

		__syncthreads();
		if( threadIdx.x == 0 ) {
			Real sum = 0;
			for( int i = 0; i < blockDim.x; i++ ) {
				sum += dotBuffer[i];
			}
			modeWeights[mode] = sum;

			rpos += paddedNumAtoms;
			if( rpos > totalRandoms ) {
				rpos -= totalRandoms;
			}
			randomPosition[blockIdx.x] = rpos;
		}
	}
}

__global__ void kFastNoise2_kernel( int numAtoms, int numModes, float4 *posq, float4 *noiseVal, float4 *velm, float4 *modes, float *modeWeights ) {
	// Load the weights into shared memory.
	extern __shared__ float weightBuffer[];
	for( int mode = threadIdx.x; mode < numModes; mode += blockDim.x ) {
		weightBuffer[mode] = modeWeights[mode];
	}
	__syncthreads();

	// Compute the projected forces and update the atom positions.
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real invMass = velm[atom].w, sqrtInvMass = sqrt( invMass );

		float3 r = make_float3( 0.0f, 0.0f, 0.0f );
		for( int mode = 0; mode < numModes; mode++ ) {
			float4 m = modes[mode * numAtoms + atom];
			float weight = weightBuffer[mode];
			r.x += m.x * weight;
			r.y += m.y * weight;
			r.z += m.z * weight;
		}

		//r.x *= sqrtInvMass;
		//r.y *= sqrtInvMass;
		//r.z *= sqrtInvMass;
		noiseVal[atom] = make_float4( noiseVal[atom].x - r.x, noiseVal[atom].y - r.y, noiseVal[atom].z - r.z, 0.0f );
		noiseVal[atom].x *= sqrtInvMass;
		noiseVal[atom].y *= sqrtInvMass;
		noiseVal[atom].z *= sqrtInvMass;

		float4 pos = posq[atom];
		pos.x += noiseVal[atom].x;
		pos.y += noiseVal[atom].y;
		pos.z += noiseVal[atom].z;
		posq[atom] = pos;
	}
}
*/

float rand_gauss (void) {
  float v1,v2,s;

    do {
        v1 = 2.0 * ((float) rand()/RAND_MAX) - 1;
	    v2 = 2.0 * ((float) rand()/RAND_MAX) - 1;

	        s = v1*v1 + v2*v2;
		  } while ( s >= 1.0 );

		    if (s == 0.0)
		        return 0.0;
			  else
			      return (v1*sqrt(-2.0 * log(s) / s));
			      }

void kFastNoise(CUmodule* module, CudaContext* cu, int numModes, float kT, int& iterations, CudaArray& modes, CudaArray& modeWeights, float maxEigenvalue, CudaArray& noiseVal, CudaArray& randomIndex, CudaArray& oldpos, float stepSize ) {
	int atoms = cu->getNumAtoms();
	int paddednumatoms = cu->getPaddedNumAtoms();
	/*int rsize = cu->getIntegrationUtilities().getRandom().getSize();
	std::vector<float4> rands(rsize);
	for (int i = 0 ; i < rsize; i++) {
		rands[i].w = rand_gauss();;
		rands[i].x = rand_gauss();;
		rands[i].y = rand_gauss();;
		rands[i].z = rand_gauss();;
	}
	cu->getIntegrationUtilities().getRandom().upload(rands);*/
	void* fastnoise1Args[] = {&atoms, &paddednumatoms, &numModes, &kT, &oldpos.getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer(),&cu->getIntegrationUtilities().getRandom().getDevicePointer(), &randomIndex.getDevicePointer(), &paddednumatoms, &maxEigenvalue, &stepSize};
        //CUmodule module = cu->createModule(CudaLTMDKernelSources::fastnoises);
	CUfunction fastnoise1Kernel, fastnoise2Kernel;
	
	//"extern \"C\" __global__ void kFastNoise1_kernel( int numAtoms, int paddedNumAtoms, int numModes, float kT, float4 *noiseVal, float4 *velm, float4 *modes,\n"
	//"                                       float *modeWeights, float4 *random, int *randomPosition, int totalRandoms, float maxEigenvalue, float stepSize ) {\n"

	
	fastnoise1Kernel = cu->getKernel(*module, "kFastNoise1_kernel");
	//cu->executeKernel(fastnoise1Kernel, fastnoise1Args, 16, 4, 16);
	cu->executeKernel(fastnoise1Kernel, fastnoise1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(float));

        void* fastnoise2Args[] = {&atoms, &numModes, &cu->getPosq().getDevicePointer(), &noiseVal.getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};

//"extern \"C\" __global__ void kFastNoise2_kernel( int numAtoms, int numModes, float4 *posq, float4 *noiseVal, float4 *velm, float4 *modes, float *modeWeights ) {\n"
        fastnoise2Kernel = cu->getKernel(*module, "kFastNoise2_kernel");
	cu->executeKernel(fastnoise2Kernel, fastnoise2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes*sizeof(float));
	//cu->executeKernel(fastnoise2Kernel, fastnoise2Args, 16, 4, 48);


	/*void* linmin1Args[] = {&atoms, &numModes, &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
        CUmodule module = cu->createModule(CudaLTMDKernelSources::linearMinimizers);
	CUfunction linmin1Kernel, linmin2Kernel;
        linmin1Kernel = cu->getKernel(module, "kFastNoise1_kernel");
	cu->executeKernel(linmin1Kernel, linmin1Args, cu->TileSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(float));
	

	kFastNoise1_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, gpu->sim.update_threads_per_block *sizeof( float ) >>> (
		gpu->natoms, gpu->sim.paddedNumberOfAtoms, numModes, gpu->sim.kT, gpu->sim.pPosqP, gpu->sim.pVelm4, modes._pDevData, modeWeights._pDevData, gpu->sim.pRandom4, gpu->sim.pRandomPosition, gpu->sim.randoms, maxEigenvalue, stepSize
	);*/

	//LAUNCHERROR( "kFastNoise1" );
	//kFastNoise2_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, numModes *sizeof( float ) >>> (
//		gpu->natoms, numModes, gpu->sim.pPosq, noiseVal._pDevData, gpu->sim.pVelm4, modes._pDevData, modeWeights._pDevData
//	);
	//LAUNCHERROR( "kFastNoise2" );

// Update randoms if necessary
// Again, assuming 20 for randomIterations because it was set to that in the construct and never changed by our integrator
//int randomIterations = 20;
//iterations++;
//if( iterations == randomIterations ) {
                                                 //kGenerateRandoms( gpu );
//                                                                 cu->getIntegrationUtilities().prepareRandomNumbers( paddednumatoms   ); // 
//                                                                                 iterations = 0;
//                                                                                         }


	// Update randoms if necessary
	/*gpu->iterations++;
	if( gpu->iterations == gpu->sim.randomIterations ) {
		kGenerateRandoms( gpu );
		gpu->iterations = 0;
	}*/
}
#endif 
/*
__global__ void kRejectMinimizationStep_kernel( int numAtoms, float4 *posq, float4 *oldPosq ) {
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		posq[atom] = oldPosq[atom];
	}
}*/

void kNMLRejectMinimizationStep(CUmodule* module, CudaContext* cu, CudaArray& oldpos ) {
        //CUmodule module = cu->createModule(CudaLTMDKernelSources::minimizationSteps);
	CUfunction rejectKernel = cu->getKernel(*module, "kRejectMinimizationStep_kernel");
	// TMC not sure how to get old positions
	int atoms = cu->getNumAtoms();
	void* rejectArgs[] = {&atoms, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer() };
	cu->executeKernel(rejectKernel, rejectArgs, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize);
	//kRejectMinimizationStep_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block >>> ( gpu->natoms, gpu->sim.pPosq, gpu->sim.pOldPosq );
	//LAUNCHERROR( "kRejectMinimizationStep" );
}
/*
__global__ void kAcceptMinimizationStep_kernel( int numAtoms, float4 *posq, float4 *oldPosq ) {
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		oldPosq[atom] = posq[atom];
	}
}*/

void kNMLAcceptMinimizationStep(CUmodule* module, CudaContext* cu, CudaArray& oldpos ) {
        //CUmodule module = cu->createModule(CudaLTMDKernelSources::minimizationSteps); // This statement takes a very long time.  WHY??  -TMC
	CUfunction acceptKernel = cu->getKernel(*module, "kAcceptMinimizationStep_kernel");
	// TMC not sure how to get old positions
	int atoms = cu->getNumAtoms();
	void* acceptArgs[] = {&atoms, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer() };
	cu->executeKernel(acceptKernel, acceptArgs, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize);
	//kAcceptMinimizationStep_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block >>> ( gpu->natoms, gpu->sim.pPosq, gpu->sim.pOldPosq );
	//LAUNCHERROR( "kAcceptMinimizationStep" );
}
/*
__global__ void kNMLLinearMinimize1_kernel( int numAtoms, int numModes, float4 *velm, float4 *force, float4 *modes, float *modeWeights ) {
	extern __shared__ float dotBuffer[];
	for( int mode = blockIdx.x; mode < numModes; mode += gridDim.x ) {
		// Compute the projection of the mass weighted force onto one normal mode vector.
		Real dot = 0.0f;
		for( int atom = threadIdx.x; atom < numAtoms; atom += blockDim.x ) {
			const Real scale = sqrt( velm[atom].w );
			const int modePos = mode * numAtoms + atom;

			float4 f = force[atom];
			float4 m = modes[modePos];

			dot += scale * ( f.x * m.x + f.y * m.y + f.z * m.z );
		}
		dotBuffer[threadIdx.x] = dot;

		__syncthreads();
		if( threadIdx.x == 0 ) {
			Real sum = 0;
			for( int i = 0; i < blockDim.x; i++ ) {
				sum += dotBuffer[i];
			}
			modeWeights[mode] = sum;
		}
	}
}

__global__ void kNMLLinearMinimize2_kernel( int numAtoms, int numModes, float invMaxEigen, float4 *posq, float4 *posqP, float4 *velm, float4 *force, float4 *modes, float *modeWeights ) {
	// Load the weights into shared memory.
	extern __shared__ float weightBuffer[];
	for( int mode = threadIdx.x; mode < numModes; mode += blockDim.x ) {
		weightBuffer[mode] = modeWeights[mode];
	}
	__syncthreads();

	// Compute the projected forces and update the atom positions.
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real invMass = velm[atom].w, sqrtInvMass = sqrt( invMass ), factor = invMass * invMaxEigen;

		float3 f = make_float3( 0.0f, 0.0f, 0.0f );
		for( int mode = 0; mode < numModes; mode++ ) {
			float4 m = modes[mode * numAtoms + atom];
			float weight = weightBuffer[mode];
			f.x += m.x * weight;
			f.y += m.y * weight;
			f.z += m.z * weight;
		}

		f.x *= sqrtInvMass;
		f.y *= sqrtInvMass;
		f.z *= sqrtInvMass;
		posqP[atom] = make_float4( force[atom].x - f.x, force[atom].y - f.y, force[atom].z - f.z, 0.0f );

		float4 pos = posq[atom];
		pos.x += factor * posqP[atom].x;
		pos.y += factor * posqP[atom].y;
		pos.z += factor * posqP[atom].z;
		posq[atom] = pos;
	}
}*/

void kNMLLinearMinimize(CUmodule* module, CudaContext* cu, int numModes, float maxEigenvalue, CudaArray& oldpos, CudaArray& modes, CudaArray& modeWeights ) {
             //printf("K LIN MIN\n");
	int atoms = cu->getNumAtoms();
	int paddedatoms = cu->getPaddedNumAtoms();
	void* linmin1Args[] = {&atoms, &paddedatoms, &numModes, &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
        //CUmodule module = cu->createModule(CudaLTMDKernelSources::linearMinimizers);
	CUfunction linmin1Kernel, linmin2Kernel;
        linmin1Kernel = cu->getKernel(*module, "kNMLLinearMinimize1_kernel");
	//int blocks = cu->TileSize;
	//int threads_per_block = (atoms + blocks - 1) / blocks; 
	//printf("ONE CALLING WITH: %d %d %d", cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(float));
	cu->executeKernel(linmin1Kernel, linmin1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(float));
	//cu->executeKernel(linmin1Kernel, linmin1Args, atoms);//16, 4, 16);
	linmin2Kernel = cu->getKernel(*module, "kNMLLinearMinimize2_kernel");
	float oneoverEig = 1.0f/maxEigenvalue;
	void* linmin2Args[] = {&atoms, &paddedatoms, &numModes, &oneoverEig, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer(), &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()}; 
	//printf("TWO CALLING WITH: %d %d %d", cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes*sizeof(float));
        cu->executeKernel(linmin2Kernel, linmin2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes*sizeof(float));
        //cu->executeKernel(linmin2Kernel, linmin2Args, atoms);//16, 4, 48);

}

/*
__global__ void kNMLQuadraticMinimize1_kernel( int numAtoms, float4 *posqP, float4 *velm, float4 *force, float *blockSlope ) {
	// Compute the slope along the minimization direction.
	extern __shared__ float slopeBuffer[];

	Real slope = 0.0f;
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real invMass = velm[atom].w;
		const float4 xp = posqP[atom];
		const float4 f = force[atom];

		slope -= invMass * ( xp.x * f.x + xp.y * f.y + xp.z * f.z );
	}
	slopeBuffer[threadIdx.x] = slope;
	__syncthreads();
	if( threadIdx.x == 0 ) {
		for( int i = 1; i <  blockDim.x; i++ ) {
			slope += slopeBuffer[i];
		}
		blockSlope[blockIdx.x] = slope;
	}
}

__global__ void kNMLQuadraticMinimize2_kernel( int numAtoms, float currentPE, float lastPE, float invMaxEigen, float4 *posq, float4 *posqP, float4 *velm, float *blockSlope, float *lambdaval ) {
	// Load the block contributions into shared memory.
	extern __shared__ float slopeBuffer[];
	for( int block = threadIdx.x; block < gridDim.x; block += blockDim.x ) {
		slopeBuffer[block] = blockSlope[block];
	}

	__syncthreads();

	// Compute the scaling coefficient.
	if( threadIdx.x == 0 ) {
		Real slope = 0.0f;
		for( int i = 0; i < gridDim.x; i++ ) {
			slope += slopeBuffer[i];
		}
		Real lambda = invMaxEigen;
		Real oldLambda = lambda;
		Real a = ( ( ( lastPE - currentPE ) / oldLambda + slope ) / oldLambda );

		if( a != 0.0f ) {
			const Real b = slope - 2.0f * a * oldLambda;
			lambda = -b / ( 2.0f * a );
		} else {
			lambda = 0.5f * oldLambda;
		}

		if( lambda <= 0.0f ) {
			lambda = 0.5f * oldLambda;
		}

		slopeBuffer[0] = lambda - oldLambda;

		// Store variables for retrival
		lambdaval[0] = lambda;
	}

	__syncthreads();

	// Remove previous position update (-oldLambda) and add new move (lambda).
	const Real dlambda = slopeBuffer[0];
	for( int atom = threadIdx.x + blockIdx.x * blockDim.x; atom < numAtoms; atom += blockDim.x * gridDim.x ) {
		const Real factor = velm[atom].w * dlambda;

		float4 pos = posq[atom];
		pos.x += factor * posqP[atom].x;
		pos.y += factor * posqP[atom].y;
		pos.z += factor * posqP[atom].z;
		posq[atom] = pos;
	}
}*/

void kNMLQuadraticMinimize(CUmodule* module, CudaContext* cu, float maxEigenvalue, float currentPE, float lastPE, CudaArray& oldpos, CudaArray& slopeBuffer, CudaArray& lambdaval ) {
	int atoms = cu->getNumAtoms();
	int paddedatoms = cu->getPaddedNumAtoms();
	void* quadmin1Args[] = {&atoms, &paddedatoms, 
				&oldpos.getDevicePointer(), 
				&cu->getVelm().getDevicePointer(), 
				&cu->getForce().getDevicePointer(), 
				&slopeBuffer.getDevicePointer()};
        //CUmodule module = cu->createModule(CudaLTMDKernelSources::quadraticMinimizers);
	CUfunction quadmin1Kernel, quadmin2Kernel;
        quadmin1Kernel = cu->getKernel(*module, "kNMLQuadraticMinimize1_kernel");
	cu->executeKernel(quadmin1Kernel, quadmin1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize*sizeof(float)); 
	//kNMLQuadraticMinimize1_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, gpu->sim.update_threads_per_block *sizeof( float ) >>> ( gpu->natoms,
	//		gpu->sim.pPosqP, gpu->sim.pVelm4, gpu->sim.pForce4, slopeBuffer._pDevData );
	//LAUNCHERROR( "kNMLQuadraticMinimize1" );

	float oneoverEig = 1.0f/maxEigenvalue;
	// TMC not sure about pPosqP
	void* quadmin2Args[] = {&atoms, &currentPE, &lastPE, &oneoverEig, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer(), &cu->getVelm().getDevicePointer(), &slopeBuffer.getDevicePointer(), &lambdaval.getDevicePointer()}; 
        quadmin2Kernel = cu->getKernel(*module, "kNMLQuadraticMinimize2_kernel");
        cu->executeKernel(quadmin2Kernel, quadmin2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->getNumThreadBlocks()*cu->ThreadBlockSize*sizeof(float)); 
	
	//kNMLQuadraticMinimize2_kernel <<< gpu->sim.blocks, gpu->sim.update_threads_per_block, gpu->sim.blocks *sizeof( float ) >>> ( gpu->natoms, currentPE,
	//		lastPE, 1.0f / maxEigenvalue, gpu->sim.pPosq, gpu->sim.pPosqP, gpu->sim.pVelm4, slopeBuffer._pDevData, lambdaval._pDevData );
	//LAUNCHERROR( "kNMLQuadraticMinimize2" );
}
