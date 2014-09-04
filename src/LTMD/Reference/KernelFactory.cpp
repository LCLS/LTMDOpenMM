#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include "LTMD/Reference/StepKernel.h"
#include "LTMD/Reference/KernelFactory.h"

#include <iostream>

namespace OpenMM {
	namespace LTMD {
		namespace Reference {
			KernelImpl *KernelFactory::createKernelImpl( std::string name, const Platform &platform, ContextImpl &context ) const {
				ReferencePlatform::PlatformData &data = *static_cast<ReferencePlatform::PlatformData *>( context.getPlatformData() );
				std::cout << "trying to create step kernel" << std::endl;
				if( name == StepKernel::Name() ) {
					return new Reference::StepKernel( name, platform, data );
				}
				std::cout << "step kernel not created" << std::endl;
				throw OpenMMException( ( std::string( "Tried to create kernel with illegal kernel name '" ) + name + "'" ).c_str() );
			}
		}
	}
}
