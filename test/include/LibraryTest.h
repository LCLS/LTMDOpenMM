#ifndef OPENMM_LTMD_LIBRARYTEST_H_
#define OPENMM_LTMD_LIBRARYTEST_H_

#include <cppunit/extensions/HelperMacros.h>

namespace LTMD {
	namespace Library {
		class Test : public CppUnit::TestFixture  {
			private:
				CPPUNIT_TEST_SUITE( Test );
				CPPUNIT_TEST( SortEigenvalue );
				CPPUNIT_TEST_SUITE_END();
			public:
				void SortEigenvalue();
		};
	}
}

#endif // OPENMM_LTMD_LIBRARYTEST_H_
