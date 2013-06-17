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

#include "OpenMM.h"
#include "CudaIntegrationUtilities.h"
#include "CudaKernels.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "openmm/internal/ContextImpl.h"
#include "CudaLTMDKernelSources.h"
#include "LTMD/Integrator.h"

#include <iostream>
using namespace std;

using namespace OpenMM;

extern void kGenerateRandoms( CudaContext* gpu );
void kNMLUpdate(CUmodule* module, CudaContext* gpu, float deltaT, float tau, float kT, int numModes, int& iterations, CudaArray& modes, CudaArray& modeWeights, CudaArray& noiseValues, CudaArray& randomIndex);
void kNMLRejectMinimizationStep(CUmodule* module, CudaContext* gpu, CudaArray& oldpos );
void kNMLAcceptMinimizationStep(CUmodule* module, CudaContext* gpu, CudaArray& oldpos );
void kNMLLinearMinimize(CUmodule* module, CudaContext* gpu, int numModes, float maxEigenvalue, CudaArray& oldpos, CudaArray& modes, CudaArray& modeWeights );
void kNMLQuadraticMinimize(CUmodule* module, CudaContext* gpu, float maxEigenvalue, float currentPE, float lastPE, CudaArray& oldpos, CudaArray& slopeBuffer, CudaArray& lambdaval );
void kFastNoise(CUmodule* module, CudaContext* cu, int numModes, float kT, int& iterations, CudaArray& modes, CudaArray& modeWeights, float maxEigenvalue, CudaArray& noiseVal, CudaArray& randomIndex, CudaArray& oldpos, float stepSize );


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
				minmodule = data.contexts[0]->createModule(CudaLTMDKernelSources::minimizationSteps);
				linmodule = data.contexts[0]->createModule(CudaLTMDKernelSources::linearMinimizers);
				quadmodule = data.contexts[0]->createModule(CudaLTMDKernelSources::quadraticMinimizers);
				fastmodule = data.contexts[0]->createModule(CudaLTMDKernelSources::fastnoises);
				updatemodule = data.contexts[0]->createModule(CudaLTMDKernelSources::NMLupdates, "-DFAST_NOISE=1");
				MinimizeLambda = new CudaArray( *(data.contexts[0]), 1, sizeof(float), "MinimizeLambda" );
				//data.contexts[0]->getPlatformData().initializeContexts(system);
				mParticles = data.contexts[0]->getNumAtoms();
				cout << "Initialize A" << endl;
			    //NoiseValues = new CUDAStream<float4>( 1, mParticles, "NoiseValues" );
			    NoiseValues = new CudaArray( *(data.contexts[0]), mParticles, sizeof(float4), "NoiseValues" );
				cout << "Initialize B" << endl;
				/*for( size_t i = 0; i < mParticles; i++ ){
					(*NoiseValues)[i] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
				}*/
				std::vector<float4> tmp(mParticles);
				cout << "Initialize C" << endl;
				for (size_t i = 0; i < mParticles; i++) {
				    tmp[i] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
                                }
				cout << "Initialize D" << endl;
				NoiseValues->upload(tmp);
				cout << "Initialize E" << endl;

				// From what I see this is no longer there, TMC
				// I could be wrong...
				//data.contexts[0]->seed = ( unsigned long ) integrator.getRandomNumberSeed();
				int seed = ( unsigned long ) integrator.getRandomNumberSeed();
				cout << "Initialize F" << endl;

				//gpuInitializeRandoms( data.gpu );
				//gpuInitializeRandoms( data.contexts[0] ); // Not sure about this, TMC
				data.contexts[0]->getIntegrationUtilities().initRandomNumberGenerator(seed);				
				cout << "Initialize G" << endl;

				// Generate a first set of randoms
				// TMC - I believe it is all done at one time
				//float4* v = new float4[mParticles]; int i;
        /*data.contexts[0]->getIntegrationUtilities().getRandom().getDevicePointer();
        data.contexts[0]->getVelm().download(v);
        for (i = 0; i < mParticles; i++) {
            printf("RANDOM BEFORE %d: %f %f %f %f\n", i, v[i].w, v[i].x, v[i].y, v[i].z);
        }*/
				randomPos = data.contexts[0]->getIntegrationUtilities().prepareRandomNumbers(data.contexts[0]->getPaddedNumAtoms());
				cout << "Initialize H" << endl;
        /*data.contexts[0]->getIntegrationUtilities().getRandom().getDevicePointer();
        data.contexts[0]->getVelm().download(v);
        for (i = 0; i < mParticles; i++) {
            printf("RANDOM AFTER %d: %f %f %f %f\n", i, v[i].w, v[i].x, v[i].y, v[i].z);
        }*/
				//kGenerateRandoms( data.gpu );
				//kGenerateRandoms( data.contexts[0] );
			}

			void StepKernel::ProjectionVectors( const Integrator &integrator ) {
				//check if projection vectors changed
				//printf("PROJ VEC A\n");
				bool modesChanged = integrator.getProjVecChanged();
				//printf("PROJ VEC B\n");

				//projection vectors changed or never allocated
				if( modesChanged || modes == NULL ) {
				//printf("PROJ VEC C\n");
					int numModes = integrator.getNumProjectionVectors();
				//printf("PROJ VEC A\n");

					//valid vectors?
					if( numModes == 0 ) {
						throw OpenMMException( "Projection vector size is zero." );
					}
				//printf("PROJ VEC D\n");

					//if( modes != NULL && modes->_length != numModes * mParticles ) {
					if( modes != NULL && modes->getSize() != numModes * mParticles ) {
						delete modes;
						delete modeWeights;
						modes = NULL;
						modeWeights = NULL;
					}
				//printf("PROJ VEC E\n");
					if( modes == NULL ) {
						/*modes = new CUDAStream<float4>( numModes * mParticles, 1, "NormalModes" );
						modeWeights = new CUDAStream<float>( numModes > data.gpu->sim.blocks ? numModes : data.gpu->sim.blocks, 1, "NormalModeWeights" );*/
						//cu->getNumThreadBlocks()*cu->ThreadBlockSize
						modes = new CudaArray( *(data.contexts[0]), numModes * mParticles, sizeof(float4), "NormalModes" );
						modeWeights = new CudaArray( *(data.contexts[0]), (numModes > data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize), sizeof(float), "NormalModeWeights" );
						oldpos = new CudaArray( *(data.contexts[0]), data.contexts[0]->getPaddedNumAtoms(), sizeof(float4), "OldPositions" );
						pPosqP = new CudaArray( *(data.contexts[0]), data.contexts[0]->getPaddedNumAtoms(), sizeof(float4), "MidIntegPositions" );
						randomIndex = new CudaArray( *(data.contexts[0]), (numModes > data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize), sizeof(int), "RandomIndices" );
				int numrandpos = (numModes > data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize);
				vector<int> tmp2(numrandpos, randomPos);
				randomIndex->upload(tmp2);
						modesChanged = true;
					}
				//printf("PROJ VEC F\n");
					if( modesChanged ) {
				//printf("PROJ VEC F1\n");
						int index = 0;
				//printf("PROJ VEC F2\n");
						const std::vector<std::vector<Vec3> >& modeVectors = integrator.getProjectionVectors();
				//printf("PROJ VEC F3\n");
				                std::vector<float4> tmp(numModes*mParticles);;
				//printf("PROJ VEC F4: %d %d %d \n", modeVectors.size(), numModes, mParticles);
						for( int i = 0; i < numModes; i++ ){
							for( int j = 0; j < mParticles; j++ ) {
								tmp[index++] = make_float4( ( float ) modeVectors[i][j][0], ( float ) modeVectors[i][j][1], ( float ) modeVectors[i][j][2], 0.0f );
								//( *modes )[index++] = make_float4( ( float ) modeVectors[i][j][0], ( float ) modeVectors[i][j][1], ( float ) modeVectors[i][j][2], 0.0f );
							}
						}
				//printf("PROJ VEC F5\n");
						//modes->Upload();
						modes->upload(tmp);
				//printf("PROJ VEC F6\n");
					}
				//printf("PROJ VEC G\n");
				}
			}

			void StepKernel::setOldPositions() {
				data.contexts[0]->getPosq().copyTo(*oldpos);
                        }

			void StepKernel::Integrate( OpenMM::ContextImpl &context, const Integrator &integrator ) {
				ProjectionVectors( integrator );

#ifdef FAST_NOISE
				// Add noise for step
				// void kFastNoise( CudaContext* cu, int numModes, float kT, int& iterations, CudaArray& modes, CUDAArray& modeWeights, float maxEigenvalue, CUDAArray& noiseVal, float stepSize )
				float mw[modeWeights->getSize()];
				int paddedatoms = data.contexts[0]->getPaddedNumAtoms();
				modeWeights->download(mw);
				printf("BEFORE ");
				for (int i = 0; i < 12; i++)
				   printf("%f ", mw[i]);
				printf("\n");
				printf("NOISESCALE: %f", sqrt(2*BOLTZ*integrator.getTemperature()*1.0f/integrator.getMaxEigenvalue()));
				float4 noise[mParticles];
				data.contexts[0]->getPosq().download(noise);
				printf("POS BEFORE: ");
				for (int i = 0; i < paddedatoms; i++)
				   printf("%f %f %f", noise[i].x, noise[i].y, noise[i].z);
				printf("\n");
				kFastNoise(&fastmodule, data.contexts[0], integrator.getNumProjectionVectors(), (float) (BOLTZ * integrator.getTemperature()), iterations, *modes, *modeWeights, integrator.getMaxEigenvalue(), *NoiseValues, *randomIndex, *pPosqP, integrator.getStepSize() );
				modeWeights->download(mw);
				printf("AFTER ");
				for (int i = 0; i < 12; i++)
				   printf("%f ", mw[i]);
				printf("\n");
				data.contexts[0]->getPosq().download(noise);
				printf("POS AFTER: ");
				for (int i = 0; i < paddedatoms; i++)
				   printf("%f %f %f", noise[i].x, noise[i].y, noise[i].z);
				printf("\n");
#endif

				// Calculate Constants
				//data.gpu->sim.deltaT = integrator.getStepSize();
				//data.gpu->sim.oneOverDeltaT = 1.0f / data.gpu->sim.deltaT;

				const double friction = integrator.getFriction();
				//data.gpu->sim.tau = friction == 0.0f ? 0.0f : 1.0f / friction;
				//data.gpu->sim.T = ( float ) integrator.getTemperature();
				//data.gpu->sim.kT = ( float )( BOLTZ * integrator.getTemperature() );

        			iterations++;
				// TMC This parameter was set by default to 20 in the old OpenMm
				// Our code does not change it, so I am assuming a value of 20.
				// If we want to change it, it should be a parameter for our integrator.
				int randomIterations = 20;
        			if( iterations == randomIterations ) {
                 			randomPos = data.contexts[0]->getIntegrationUtilities().prepareRandomNumbers( data.contexts[0]->getPaddedNumAtoms()  );
					int numModes = integrator.getNumProjectionVectors();;
				        int numrandpos = (numModes > data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize);
				       vector<int> tmp2(numrandpos, randomPos);
				        randomIndex->upload(tmp2);
               				iterations = 0;
        			}
        context.updateContextState();        
        /*long long v[576*3];
	int i;
        data.contexts[0]->getForce().download(v);
        for (i = 0; i < 576*3; i++) {
            printf("FORCE BEFORE FUNCTIONA CALL %d: %f\n", i, (double)v[i]/(double)0x100000000);
        }*/
	//delete v;
				// Do Step
				kNMLUpdate(&updatemodule,
					data.contexts[0], 
					   integrator.getStepSize(),
					   friction == 0.0f ? 0.0f : 1.0f / friction,
					   (float) (BOLTZ * integrator.getTemperature()),
					   integrator.getNumProjectionVectors(), kIterations, *modes, *modeWeights, *NoiseValues, *randomIndex  ); // TMC setting parameters for this
        			iterations++;
				// TMC This parameter was set by default to 20 in the old OpenMm
				// Our code does not change it, so I am assuming a value of 20.
				// If we want to change it, it should be a parameter for our integrator.
        			if( iterations == randomIterations ) {
                 			randomPos = data.contexts[0]->getIntegrationUtilities().prepareRandomNumbers( data.contexts[0]->getPaddedNumAtoms()  );
					int numModes = integrator.getNumProjectionVectors();;
				        int numrandpos = (numModes > data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks()*data.contexts[0]->ThreadBlockSize);
				       vector<int> tmp2(numrandpos, randomPos);
				        randomIndex->upload(tmp2);
               				iterations = 0;
        			}
				//kNMLUpdate( data.gpu, integrator.getNumProjectionVectors(), *modes, *modeWeights, *NoiseValues );
			}

			void StepKernel::UpdateTime( const Integrator &integrator ) {
				data.time += integrator.getStepSize();
				data.stepCount++;
			}

			void StepKernel::AcceptStep( OpenMM::ContextImpl &context ) {
				kNMLAcceptMinimizationStep(&minmodule, data.contexts[0], *oldpos );
			}

			void StepKernel::RejectStep( OpenMM::ContextImpl &context ) {
				kNMLRejectMinimizationStep(&minmodule, data.contexts[0], *oldpos );
			}

			void StepKernel::LinearMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy ) {
				ProjectionVectors( integrator );

				lastPE = energy;
				kNMLLinearMinimize(&linmodule, data.contexts[0], integrator.getNumProjectionVectors(), integrator.getMaxEigenvalue(), *pPosqP, *modes, *modeWeights );
			}

			double StepKernel::QuadraticMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy ) {
				ProjectionVectors( integrator );

				kNMLQuadraticMinimize(&quadmodule, data.contexts[0], integrator.getMaxEigenvalue(), energy, lastPE, *pPosqP, *modeWeights, *MinimizeLambda );
				std::vector<float> tmp;
				tmp.resize(1);
				printf("READY TO DOWNLOAD\n");
				MinimizeLambda->download(tmp);

				//return (*MinimizeLambda)[0];
				return tmp[0];
			}

		}
	}
}
