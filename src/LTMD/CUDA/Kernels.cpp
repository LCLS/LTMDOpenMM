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

#include <cmath>

#include "OpenMM.h"
#include "CudaKernels.h"
#include "openmm/internal/ContextImpl.h"

#include "LTMD/Integrator.h"
#include "LTMD/CUDA/Kernels.h"

extern void kGenerateRandoms( gpuContext gpu );
void kNMLUpdate( gpuContext gpu, int numModes, CUDAStream<float4>& modes, CUDAStream<float>& modeWeights );
void kNMLRejectMinimizationStep( gpuContext gpu, CUDAStream<float>& minimizerScale );
void kNMLAcceptMinimizationStep( gpuContext gpu, CUDAStream<float>& minimizerScale );
void kNMLLinearMinimize( gpuContext gpu, int numModes, float maxEigenvalue, CUDAStream<float4>& modes, CUDAStream<float>& modeWeights, CUDAStream<float>& minimizerScale );
void kNMLQuadraticMinimize( gpuContext gpu, float maxEigenvalue, float currentPE, float lastPE, CUDAStream<float>& slopeBuffer );

namespace OpenMM {
	namespace LTMD {
		namespace CUDA {
			StepKernel::StepKernel( std::string name, const Platform &platform, CudaPlatform::PlatformData &data ) : LTMD::StepKernel( name, platform ),
				data( data ), modes( NULL ), modeWeights( NULL ), minimizerScale( NULL ) {
			}

			StepKernel::~StepKernel() {
				if( modes != NULL ) {
					delete modes;
				}
				if( modeWeights != NULL ) {
					delete modeWeights;
				}
				if( minimizerScale != NULL ) {
					delete minimizerScale;
				}
			}

			void StepKernel::initialize( const System &system, const Integrator &integrator ) {
				OpenMM::cudaOpenMMInitializeIntegration( system, data, integrator );
				_gpuContext *gpu = data.gpu;
				gpu->seed = ( unsigned long ) integrator.getRandomNumberSeed();
				gpuInitializeRandoms( gpu );
				minimizerScale = new CUDAStream<float>( 1, 1, "MinimizerScale" );
			}


			void StepKernel::execute( ContextImpl &context, const Integrator &integrator, const double currentPE, const int stepType ) {
				_gpuContext *gpu = data.gpu;

				//get standard data
				int numAtoms = context.getSystem().getNumParticles();
				int numModes = integrator.getNumProjectionVectors();
				double dt = integrator.getStepSize();

				//check if projection vectors changed
				bool modesChanged = integrator.getProjVecChanged();

				//projection vectors changed or never allocated
				if( modesChanged || modes == NULL ) {
					//valid vectors?
					if( numModes == 0 ) {
						throw OpenMMException( "Projection vector size is zero." );
					}

					if( modes != NULL && modes->_length != numModes * numAtoms ) {
						delete modes;
						delete modeWeights;
						modes = NULL;
						modeWeights = NULL;
					}
					if( modes == NULL ) {
						modes = new CUDAStream<float4>( numModes * numAtoms, 1, "NormalModes" );
						modeWeights = new CUDAStream<float>( numModes > gpu->sim.blocks ? numModes : gpu->sim.blocks, 1, "NormalModeWeights" );
						modesChanged = true;
					}
					if( modesChanged ) {
						int index = 0;
						const std::vector<std::vector<Vec3> >& modeVectors = integrator.getProjectionVectors();
						for( int i = 0; i < numModes; i++ )
							for( int j = 0; j < numAtoms; j++ ) {
								( *modes )[index++] = make_float4( ( float ) modeVectors[i][j][0], ( float ) modeVectors[i][j][1], ( float ) modeVectors[i][j][2], 0.0f );
							}
						modes->Upload();
					}
					gpu->sim.deltaT = ( float ) dt;
					gpu->sim.oneOverDeltaT = ( float )( 1.0 / dt );
					double friction = integrator.getFriction();
					gpu->sim.tau = ( float )( friction == 0.0 ? 0.0 : 1.0 / friction );
					gpu->sim.T = ( float ) integrator.getTemperature();
					gpu->sim.kT = ( float )( BOLTZ * integrator.getTemperature() );
					kGenerateRandoms( gpu );
				}

				switch( stepType ) {
					case 1:
						kNMLUpdate( gpu, numModes, *modes, *modeWeights );
						break;
					case 2:
						data.time += dt;
						data.stepCount++;
						break;
					case 3:
						lastPE = currentPE;
						kNMLLinearMinimize( gpu, numModes, integrator.getMaxEigenvalue(), *modes, *modeWeights, *minimizerScale );
						break;
					case 4:
						kNMLQuadraticMinimize( gpu, integrator.getMaxEigenvalue(), currentPE, lastPE, *modeWeights );
						break;
					case 5:
						kNMLRejectMinimizationStep( gpu, *minimizerScale );
						break;
					case 6:
						kNMLAcceptMinimizationStep( gpu, *minimizerScale );
						break;
				}

				if( data.removeCM && data.stepCount % data.cmMotionFrequency == 0 ) {
					gpu->bCalculateCM = true;
				}
			}
		}
	}
}
