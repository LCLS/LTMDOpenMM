#ifndef OPENMM_LTMD_MATHTEST_H_
#define OPENMM_LTMD_MATHTEST_H_

#include <cppunit/extensions/HelperMacros.h>

namespace LTMD {
	namespace Math {
		class Test : public CppUnit::TestFixture  {
			private:
				CPPUNIT_TEST_SUITE( Test );
				CPPUNIT_TEST( Eigenvalues );
				CPPUNIT_TEST( MatrixMultiplyTest );
				CPPUNIT_TEST_SUITE_END();
			public:
				void Eigenvalues();
				void MatrixMultiplyTest();
		};
	}
}

#endif // OPENMM_LTMD_MATHTEST_H_
