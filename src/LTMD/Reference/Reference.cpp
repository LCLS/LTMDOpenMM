#include <cstdio>
#include "OpenMM.h"
#include "LTMD/Reference/KernelFactory.h"
#include "LTMD/StepKernel.h"
#include "openmm/internal/windowsExport.h"

using namespace OpenMM;

extern "C" void registerPlatforms() {

}

extern "C" void registerKernelFactories() {
	printf( "LTMD looking for reference plugin...\n" );
	try {
		Platform &platform = Platform::getPlatformByName( "Reference" );
		printf( "LTMD found reference platform... \n" );
		platform.registerKernelFactory( LTMD::StepKernel::Name(), new LTMD::Reference::KernelFactory() );
		printf( "Registered LTMD reference plugin... \n" );
	} catch( const std::exception &exc ) {
		printf( "LTMD Reference platform not found. %s", exc.what() );
	}
}
