#ifndef OPENMM_LTMD_KERNELFACTORY_H_
#define OPENMM_LTMD_KERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {
	namespace LTMD {
		namespace Reference {
			class KernelFactory : public OpenMM::KernelFactory {
				public:
					OpenMM::KernelImpl *createKernelImpl( std::string name, const OpenMM::Platform &platform, OpenMM::ContextImpl &context ) const;
			};
		}
	}
}

#endif // OPENMM_LTMD_KERNELFACTORY_H_
