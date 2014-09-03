/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2009 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
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

#include "LTMD/CUDA/StepKernel.h"

#include <cmath>
#include "SimTKOpenMMUtilities.h"
#include "OpenMM.h"
#include "CudaIntegrationUtilities.h"
#include "CudaKernels.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "openmm/internal/ContextImpl.h"
#include "CudaLTMDKernelSources.h"
#include "LTMD/Integrator.h"
#include <stdlib.h>
#include <iostream>
using namespace std;

using namespace OpenMM;

extern void kGenerateRandoms( CudaContext *gpu );
void kNMLUpdate( CUmodule *module, CudaContext *gpu, float deltaT, float tau, float kT, int numModes, int &iterations, CudaArray &modes, CudaArray &modeWeights, CudaArray &noiseValues, CudaArray &randomIndex );
void kNMLRejectMinimizationStep( CUmodule *module, CudaContext *gpu, CudaArray &oldpos );
void kNMLAcceptMinimizationStep( CUmodule *module, CudaContext *gpu, CudaArray &oldpos );
void kNMLLinearMinimize( CUmodule *module, CudaContext *gpu, int numModes, float maxEigenvalue, CudaArray &oldpos, CudaArray &modes, CudaArray &modeWeights );
void kNMLQuadraticMinimize( CUmodule *module, CudaContext *gpu, float maxEigenvalue, float currentPE, float lastPE, CudaArray &oldpos, CudaArray &slopeBuffer, CudaArray &lambdaval );
void kFastNoise( CUmodule *module, CudaContext *cu, int numModes, float kT, int &iterations, CudaArray &modes, CudaArray &modeWeights, float maxEigenvalue, CudaArray &noiseVal, CudaArray &randomIndex, CudaArray &oldpos, float stepSize );

double drand() { /* uniform distribution, (0..1] */
	return ( rand() + 1.0 ) / ( RAND_MAX + 1.0 );
}
double random_normal() { /* normal distribution, centered on 0, std dev 1 */
	return sqrt( -2 * log( drand() ) ) * cos( 2 * M_PI * drand() );
}


namespace OpenMM {
	namespace LTMD {
		namespace CUDA {
			StepKernel::StepKernel( std::string name, const Platform &platform, CudaPlatform::PlatformData &data ) : LTMD::StepKernel( name, platform ),
				data( data ), modes( NULL ), modeWeights( NULL ), minimizerScale( NULL ), MinimizeLambda( 0 ) {

				//MinimizeLambda = new CUDAStream<float>( 1, 1, "MinimizeLambda" );
				//MinimizeLambda = new CudaArray( *(data.contexts[0]), 1, sizeof(float), "MinimizeLambda" );
				iterations = 0;
				kIterations = 0;
			}

			StepKernel::~StepKernel() {
				if( modes != NULL ) {
					delete modes;
				}
				if( modeWeights != NULL ) {
					delete modeWeights;
				}
			}

			void StepKernel::initialize( const System &system, const Integrator &integrator ) {
				// TMC This is done automatically when you setup a context now.
				//OpenMM::cudaOpenMMInitializeIntegration( system, data, integrator ); // TMC not sure how to replace
				data.contexts[0]->initialize();
				minmodule = data.contexts[0]->createModule( CudaLTMDKernelSources::minimizationSteps );
				linmodule = data.contexts[0]->createModule( CudaLTMDKernelSources::linearMinimizers );
				quadmodule = data.contexts[0]->createModule( CudaLTMDKernelSources::quadraticMinimizers );
#ifdef FAST_NOISE
				fastmodule = data.contexts[0]->createModule( CudaLTMDKernelSources::fastnoises, "-DFAST_NOISE=1" );
				updatemodule = data.contexts[0]->createModule( CudaLTMDKernelSources::NMLupdates, "-DFAST_NOISE=1" );
#else
				updatemodule = data.contexts[0]->createModule( CudaLTMDKernelSources::NMLupdates, "-DFAST_NOISE=0" );
#endif

				MinimizeLambda = new CudaArray( *( data.contexts[0] ), 1, sizeof( float ), "MinimizeLambda" );
				//data.contexts[0]->getPlatformData().initializeContexts(system);
				mParticles = data.contexts[0]->getNumAtoms();
				//NoiseValues = new CUDAStream<float4>( 1, mParticles, "NoiseValues" );
				NoiseValues = new CudaArray( *( data.contexts[0] ), mParticles, sizeof( float4 ), "NoiseValues" );
				/*for( size_t i = 0; i < mParticles; i++ ){
					(*NoiseValues)[i] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
				}*/
				std::vector<float4> tmp( mParticles );
				for( size_t i = 0; i < mParticles; i++ ) {
					tmp[i] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
				}
				NoiseValues->upload( tmp );

			    data.contexts[0]->getIntegrationUtilities().initRandomNumberGenerator(integrator.getRandomNumberSeed());
			}

			void StepKernel::ProjectionVectors( const Integrator &integrator ) {
				//check if projection vectors changed
				bool modesChanged = integrator.getProjVecChanged();

				//projection vectors changed or never allocated
				if( modesChanged || modes == NULL ) {
					int numModes = integrator.getNumProjectionVectors();

					//valid vectors?
					if( numModes == 0 ) {
						throw OpenMMException( "Projection vector size is zero." );
					}

					//if( modes != NULL && modes->_length != numModes * mParticles ) {
					if( modes != NULL && modes->getSize() != numModes * mParticles ) {
						delete modes;
						delete modeWeights;
						modes = NULL;
						modeWeights = NULL;
					}
					if( modes == NULL ) {
						/*modes = new CUDAStream<float4>( numModes * mParticles, 1, "NormalModes" );
						modeWeights = new CUDAStream<float>( numModes > data.gpu->sim.blocks ? numModes : data.gpu->sim.blocks, 1, "NormalModeWeights" );*/
						//cu->getNumThreadBlocks()*cu->ThreadBlockSize
						modes = new CudaArray( *( data.contexts[0] ), numModes * mParticles, sizeof( float4 ), "NormalModes" );
						modeWeights = new CudaArray( *( data.contexts[0] ), ( numModes > data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize ), sizeof( float ), "NormalModeWeights" );
						oldpos = new CudaArray( *( data.contexts[0] ), data.contexts[0]->getPaddedNumAtoms(), sizeof( float4 ), "OldPositions" );
						pPosqP = new CudaArray( *( data.contexts[0] ), data.contexts[0]->getPaddedNumAtoms(), sizeof( float4 ), "MidIntegPositions" );
						randomIndex = new CudaArray( *( data.contexts[0] ), ( numModes > data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize ), sizeof( int ), "RandomIndices" );
						int numrandpos = ( numModes > data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize );
						vector<int> tmp2( numrandpos, randomPos );
						randomIndex->upload( tmp2 );
						modesChanged = true;
					}
					if( modesChanged ) {
						int index = 0;
						const std::vector<std::vector<Vec3> > &modeVectors = integrator.getProjectionVectors();
						std::vector<float4> tmp( numModes * mParticles );;
						for( int i = 0; i < numModes; i++ ) {
							for( int j = 0; j < mParticles; j++ ) {
								tmp[index++] = make_float4( ( float ) modeVectors[i][j][0], ( float ) modeVectors[i][j][1], ( float ) modeVectors[i][j][2], 0.0f );
							}
						}
						modes->upload( tmp );
					}
				}
			}

			void StepKernel::setOldPositions() {
				data.contexts[0]->getPosq().copyTo( *oldpos );
			}

			void StepKernel::Integrate( OpenMM::ContextImpl &context, const Integrator &integrator ) {
				ProjectionVectors( integrator );

#ifdef FAST_NOISE
				// Add noise for step
				kFastNoise( &fastmodule, data.contexts[0], integrator.getNumProjectionVectors(), ( float )( BOLTZ * integrator.getTemperature() ), iterations, *modes, *modeWeights, integrator.getMaxEigenvalue(), *NoiseValues, *randomIndex, *pPosqP, integrator.getStepSize() );
#endif

				// Calculate Constants

				const double friction = integrator.getFriction();

				iterations++;
				// TMC This parameter was set by default to 20 in the old OpenMm
				// Our code does not change it, so I am assuming a value of 20.
				// If we want to change it, it should be a parameter for our integrator.
				int randomIterations = 20;
				if( iterations == randomIterations ) {
					//randomPos = data.contexts[0]->getIntegrationUtilities().prepareRandomNumbers(data.contexts[0]->getPaddedNumAtoms());
					int paddednumatoms = data.contexts[0]->getPaddedNumAtoms();
					std::vector<float4> tmp2( paddednumatoms * 32 );
					for( size_t i = 0; i < 32 * paddednumatoms; i++ ) {
						//tmp2[i] = make_float4(random_normal(), random_normal(), random_normal(), random_normal());
						tmp2[i] = make_float4( SimTKOpenMMUtilities::getNormallyDistributedRandomNumber(),
											   SimTKOpenMMUtilities::getNormallyDistributedRandomNumber(),
											   SimTKOpenMMUtilities::getNormallyDistributedRandomNumber(),
											   SimTKOpenMMUtilities::getNormallyDistributedRandomNumber() );
					}
					randomPos = paddednumatoms; // OpenMM did this as well
					data.contexts[0]->getIntegrationUtilities().getRandom().upload( tmp2 );
					//	randomPos = data.contexts[0]->getIntegrationUtilities().prepareRandomNumbers( data.contexts[0]->getPaddedNumAtoms()  );
					int numModes = integrator.getNumProjectionVectors();;
					int numrandpos = ( numModes > data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize );
					vector<int> tmp3( numrandpos, randomPos );
					randomIndex->upload( tmp3 );
					iterations = 0;
				}
				context.updateContextState();
				// Do Step
				kNMLUpdate( &updatemodule,
							data.contexts[0],
							integrator.getStepSize(),
							friction == 0.0f ? 0.0f : 1.0f / friction,
							( float )( BOLTZ * integrator.getTemperature() ),
							integrator.getNumProjectionVectors(), kIterations, *modes, *modeWeights, *NoiseValues, *randomIndex );  // TMC setting parameters for this
				iterations++;
				// TMC This parameter was set by default to 20 in the old OpenMm
				// Our code does not change it, so I am assuming a value of 20.
				// If we want to change it, it should be a parameter for our integrator.
				if( iterations == randomIterations ) {
					//randomPos = data.contexts[0]->getIntegrationUtilities().prepareRandomNumbers(data.contexts[0]->getPaddedNumAtoms());
					int paddednumatoms = data.contexts[0]->getPaddedNumAtoms();
					std::vector<float4> tmp2( paddednumatoms * 32 );
					for( size_t i = 0; i < 32 * paddednumatoms; i++ ) {
						//tmp2[i] = make_float4(random_normal(), random_normal(), random_normal(), random_normal());
						tmp2[i] = make_float4( SimTKOpenMMUtilities::getNormallyDistributedRandomNumber(),
											   SimTKOpenMMUtilities::getNormallyDistributedRandomNumber(),
											   SimTKOpenMMUtilities::getNormallyDistributedRandomNumber(),
											   SimTKOpenMMUtilities::getNormallyDistributedRandomNumber() );
					}
					randomPos = paddednumatoms; // OpenMM did this as well
					data.contexts[0]->getIntegrationUtilities().getRandom().upload( tmp2 );
					//randomPos = data.contexts[0]->getIntegrationUtilities().prepareRandomNumbers( data.contexts[0]->getPaddedNumAtoms()  );
					int numModes = integrator.getNumProjectionVectors();;
					int numrandpos = ( numModes > data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize );
					vector<int> tmp3( numrandpos, randomPos );
					randomIndex->upload( tmp3 );
					iterations = 0;
				}
			}

			void StepKernel::UpdateTime( const Integrator &integrator ) {
				data.time += integrator.getStepSize();
				data.stepCount++;
			}

			void StepKernel::AcceptStep( OpenMM::ContextImpl &context ) {
				kNMLAcceptMinimizationStep( &minmodule, data.contexts[0], *oldpos );
			}

			void StepKernel::RejectStep( OpenMM::ContextImpl &context ) {
				kNMLRejectMinimizationStep( &minmodule, data.contexts[0], *oldpos );
			}

			void StepKernel::LinearMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy ) {
				ProjectionVectors( integrator );

				lastPE = energy;
				kNMLLinearMinimize( &linmodule, data.contexts[0], integrator.getNumProjectionVectors(), integrator.getMaxEigenvalue(), *pPosqP, *modes, *modeWeights );
			}

			double StepKernel::QuadraticMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy ) {
				ProjectionVectors( integrator );

				kNMLQuadraticMinimize( &quadmodule, data.contexts[0], integrator.getMaxEigenvalue(), energy, lastPE, *pPosqP, *modeWeights, *MinimizeLambda );
				std::vector<float> tmp;
				tmp.resize( 1 );
				printf( "READY TO DOWNLOAD\n" );
				MinimizeLambda->download( tmp );

				//return (*MinimizeLambda)[0];
				return tmp[0];
			}

		}
	}
}
