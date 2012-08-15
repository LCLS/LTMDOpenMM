#ifndef OPENMM_LTMD_LIBRARYTEST_H_
#define OPENMM_LTMD_LIBRARYTEST_H_

#include <cppunit/extensions/HelperMacros.h>

namespace LTMD {
	namespace Analysis {
		class Test : public CppUnit::TestFixture  {
			private:
				CPPUNIT_TEST_SUITE( Test );
                CPPUNIT_TEST( BlockDiagonalize );
                CPPUNIT_TEST( GeometricDOF );
				CPPUNIT_TEST_SUITE_END();
			public:
				void BlockDiagonalize();
                void GeometricDOF();
		};
	}
}

#endif // OPENMM_LTMD_LIBRARYTEST_H_
