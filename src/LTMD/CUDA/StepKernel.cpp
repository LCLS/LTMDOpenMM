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
#include "CudaKernels.h"
#include "openmm/internal/ContextImpl.h"

#include "LTMD/Integrator.h"


extern void kGenerateRandoms( gpuContext gpu );
void kNMLUpdate( gpuContext gpu, int numModes, CUDAStream<float4>& modes, CUDAStream<float>& modeWeights, CUDAStream<float4>& noiseValues );
void kNMLRejectMinimizationStep( gpuContext gpu );
void kNMLAcceptMinimizationStep( gpuContext gpu );
void kNMLLinearMinimize( gpuContext gpu, int numModes, float maxEigenvalue, CUDAStream<float4>& modes, CUDAStream<float>& modeWeights );
void kNMLQuadraticMinimize( gpuContext gpu, float maxEigenvalue, float currentPE, float lastPE, CUDAStream<float>& slopeBuffer, CUDAStream<float>& lambdaval );
void kFastNoise( gpuContext gpu, int numModes, CUDAStream<float4>& modes, CUDAStream<float>& modeWeights, float maxEigenvalue, CUDAStream<float4>& noiseValues, float stepSize );

namespace OpenMM {
	namespace LTMD {
		namespace CUDA {
			StepKernel::StepKernel( std::string name, const Platform &platform, CudaPlatform::PlatformData &data ) : LTMD::StepKernel( name, platform ),
				data( data ), modes( NULL ), modeWeights( NULL ), minimizerScale( NULL ), MinimizeLambda( 0 ) {

				MinimizeLambda = new CUDAStream<float>( 1, 1, "MinimizeLambda" );
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
				OpenMM::cudaOpenMMInitializeIntegration( system, data, integrator );

				mParticles = system.getNumParticles();
			    NoiseValues = new CUDAStream<float4>( 1, mParticles, "NoiseValues" );

				for( size_t i = 0; i < mParticles; i++ ){
					(*NoiseValues)[i] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
				}
				NoiseValues->Upload();


				data.gpu->seed = ( unsigned long ) integrator.getRandomNumberSeed();
				gpuInitializeRandoms( data.gpu );

				// Generate a first set of randoms
				kGenerateRandoms( data.gpu );
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

					if( modes != NULL && modes->_length != numModes * mParticles ) {
						delete modes;
						delete modeWeights;
						modes = NULL;
						modeWeights = NULL;
					}
					if( modes == NULL ) {
						modes = new CUDAStream<float4>( numModes * mParticles, 1, "NormalModes" );
						modeWeights = new CUDAStream<float>( numModes > data.gpu->sim.blocks ? numModes : data.gpu->sim.blocks, 1, "NormalModeWeights" );
						modesChanged = true;
					}
					if( modesChanged ) {
						int index = 0;
						const std::vector<std::vector<Vec3> >& modeVectors = integrator.getProjectionVectors();
						for( int i = 0; i < numModes; i++ ){
							for( int j = 0; j < mParticles; j++ ) {
								( *modes )[index++] = make_float4( ( float ) modeVectors[i][j][0], ( float ) modeVectors[i][j][1], ( float ) modeVectors[i][j][2], 0.0f );
							}
						}
						modes->Upload();
					}
				}
			}

			void StepKernel::Integrate( OpenMM::ContextImpl &context, const Integrator &integrator ) {
				ProjectionVectors( integrator );

#ifdef FAST_NOISE
				// Add noise for step
				kFastNoise( data.gpu, integrator.getNumProjectionVectors(), *modes, *modeWeights, integrator.getMaxEigenvalue(), *NoiseValues, integrator.getStepSize() );
#endif

				// Calculate Constants
				data.gpu->sim.deltaT = integrator.getStepSize();
				data.gpu->sim.oneOverDeltaT = 1.0f / data.gpu->sim.deltaT;

				const double friction = integrator.getFriction();
				data.gpu->sim.tau = friction == 0.0f ? 0.0f : 1.0f / friction;
				data.gpu->sim.T = ( float ) integrator.getTemperature();
				data.gpu->sim.kT = ( float )( BOLTZ * integrator.getTemperature() );

				// Do Step
				kNMLUpdate( data.gpu, integrator.getNumProjectionVectors(), *modes, *modeWeights, *NoiseValues );
			}

			void StepKernel::UpdateTime( const Integrator &integrator ) {
				data.time += integrator.getStepSize();
				data.stepCount++;
			}

			void StepKernel::AcceptStep( OpenMM::ContextImpl &context ) {
				kNMLAcceptMinimizationStep( data.gpu );
			}

			void StepKernel::RejectStep( OpenMM::ContextImpl &context ) {
				kNMLRejectMinimizationStep( data.gpu );
			}

			void StepKernel::LinearMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy ) {
				ProjectionVectors( integrator );

				lastPE = energy;
				kNMLLinearMinimize( data.gpu, integrator.getNumProjectionVectors(), integrator.getMaxEigenvalue(), *modes, *modeWeights );
			}

			double StepKernel::QuadraticMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy ) {
				ProjectionVectors( integrator );

				kNMLQuadraticMinimize( data.gpu, integrator.getMaxEigenvalue(), energy, lastPE, *modeWeights, *MinimizeLambda );

				MinimizeLambda->Download();

				return (*MinimizeLambda)[0];
			}
		}
	}
}
