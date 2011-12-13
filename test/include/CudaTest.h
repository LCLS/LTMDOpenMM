#ifndef OPENMM_LTMD_CUDA_H_
#define OPENMM_LTMD_CUDA_H_

#include <cppunit/extensions/HelperMacros.h>

namespace LTMD {
	namespace Cuda {
		class Test : public CppUnit::TestFixture  {
			private:
				CPPUNIT_TEST_SUITE( Test );
					CPPUNIT_TEST( Initialisation );
					CPPUNIT_TEST( Projection );
					CPPUNIT_TEST( MinimizeIntegration );
				CPPUNIT_TEST_SUITE_END();
			public:
				void Initialisation();
				void Projection();
				void MinimizeIntegration();
		};
	}
}

#endif // OPENMM_LTMD_CUDA_H_
