#include <iostream>

#include "LTMD/CUDA/StepKernel.h"
#include "LTMD/CUDA/KernelFactory.h"

#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;
using namespace std;

namespace OpenMM {
	namespace LTMD {
		namespace CUDA {
			KernelImpl *KernelFactory::createKernelImpl( std::string name, const Platform &platform, ContextImpl &context ) const {
				CudaPlatform::PlatformData &data = *static_cast<CudaPlatform::PlatformData *>( context.getPlatformData() );

				if( name != StepKernel::Name() ) {
					throw OpenMMException( ( std::string( "Tried to create kernel with illegal kernel name '" ) + name + "'" ).c_str() );
				}

				return new StepKernel( name, platform, data );
			}
		}
	}
}
