#ifndef OPENMM_LTMD_CUDA_KERNELFACTORY_H_
#define OPENMM_LTMD_CUDA_KERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {
	namespace LTMD {
		namespace CUDA {
			class KernelFactory : public OpenMM::KernelFactory {
				public:
					KernelImpl *createKernelImpl( std::string name, const Platform &platform, ContextImpl &context ) const;
			};
		}
	}
}

#endif // OPENMM_LTMD_CUDA_KERNELFACTORY_H_
