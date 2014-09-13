#include <cstdio>
#include "OpenMM.h"
#include "openmm/internal/windowsExport.h"

#include "LTMD/CUDA/KernelFactory.h"
#include "LTMD/StepKernel.h"

using namespace OpenMM;

extern "C" void registerPlatforms() {

}

extern "C" void registerKernelFactories() {
	printf( "LTMD looking for CUDA plugin...\n" );
	try {
		Platform &platform = Platform::getPlatformByName( "CUDA" );
		printf( "LTMD found CUDA platform...\n" );
		platform.registerKernelFactory( LTMD::StepKernel::Name(), new LTMD::CUDA::KernelFactory() );
		printf( "LTMD registered CUDA plugin ... \n" );
	} catch( const std::exception &exc ) {
		printf( "LTMD CUDA platform not found. %s\n", exc.what() );
	}
}
