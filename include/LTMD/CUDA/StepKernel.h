#ifndef OPENMM_LTMD_CUDA_KERNELS_H_
#define OPENMM_LTMD_CUDA_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2010 Stanford University and the Authors.      *
 * Authors: Chris Sweet                                                       *
 * Contributors: Christopher Bruns                                            *
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

#include "LTMD/StepKernel.h"

#include "CudaPlatform.h"
//#include "kernels/gputypes.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "CudaIntegrationUtilities.h"

static const float KILO                     =    1e3;                      // Thousand
static const float BOLTZMANN                =    1.380658e-23f;            // (J/K)
static const float AVOGADRO                 =    6.0221367e23f;            // ()
static const float RGAS                     =    BOLTZMANN * AVOGADRO;     // (J/(mol K))
static const float BOLTZ                    = ( RGAS / KILO );             // (kJ/(mol K))

namespace OpenMM {
	namespace LTMD {
		namespace CUDA {
			class StepKernel : public LTMD::StepKernel {
				public:
					StepKernel( std::string name, const OpenMM::Platform &platform, OpenMM::CudaPlatform::PlatformData &data );
					~StepKernel();
					/**
					 * Initialize the kernel, setting up the particle masses.
					 *
					 * @param system     the System this kernel will be applied to
					 * @param integrator the LangevinIntegrator this kernel will be used for
					 */
					void initialize( const OpenMM::System &system, const Integrator &integrator );

					void Integrate( OpenMM::ContextImpl &context, const Integrator &integrator );
					void UpdateTime( const Integrator &integrator );
					void setOldPositions( );
					void AcceptStep( OpenMM::ContextImpl &context );
					void RejectStep( OpenMM::ContextImpl &context );

					void LinearMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy );
					double QuadraticMinimize( OpenMM::ContextImpl &context, const Integrator &integrator, const double energy );
					virtual double computeKineticEnergy( OpenMM::ContextImpl &context, const Integrator &integrator ) {

						//return context.calcKineticEnergy();
						return data.contexts[0]->getIntegrationUtilities().computeKineticEnergy( 0.5 * integrator.getStepSize() );
						//return ((CudaContext&)context).getIntegrationUtilities().computeKineticEnergy(0.5*integrator.getStepSize());
						//return context.getIntegrationUtilities().computeKineticEnergy(0.5*integrator.getStepSize());
					}


				private:
					void ProjectionVectors( const Integrator &integrator );
				private:
					unsigned int mParticles;
					OpenMM::CudaPlatform::PlatformData &data;
					CudaArray *modes, *NoiseValues;
					CudaArray *modeWeights;
					CudaArray *randomIndex;
					CudaArray *minimizerScale;
					CudaArray *MinimizeLambda;
					CudaArray *oldpos;
					CudaArray *pPosqP;
					//CUDAStream<float4> *modes, *NoiseValues;
					//CUDAStream<float>* modeWeights;
					//CUDAStream<float>* minimizerScale;
					//CUDAStream<float>* MinimizeLambda;
					double lastPE;
					double prevTemp, prevFriction, prevStepSize;
					int iterations;
					int kIterations;
					int randomPos;
					CUmodule minmodule;
					CUmodule linmodule;
					CUmodule quadmodule;
					CUmodule fastmodule;
					CUmodule updatemodule;
			};
		}
	}
}

#endif // OPENMM_LTMD_CUDA_KERNELS_H_
