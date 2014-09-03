#ifndef OPENMM_LTMD_MATHTEST_H_
#define OPENMM_LTMD_MATHTEST_H_

#include <cppunit/extensions/HelperMacros.h>

namespace LTMD {
	namespace Math {
		class Test : public CppUnit::TestFixture  {
			private:
				CPPUNIT_TEST_SUITE( Test );
				CPPUNIT_TEST( EigenvalueTest );
				CPPUNIT_TEST( EigenvectorTest );
				CPPUNIT_TEST( MatrixMultiplyTest );
				CPPUNIT_TEST( TransposeMatrixMultiplyTest );
				CPPUNIT_TEST( TransposeAMatrixMultiplyTest );
				CPPUNIT_TEST( TransposeBMatrixMultiplyTest );
				CPPUNIT_TEST_SUITE_END();
			public:
				void EigenvalueTest();
				void EigenvectorTest();
				void MatrixMultiplyTest();
				void TransposeMatrixMultiplyTest();
				void TransposeAMatrixMultiplyTest();
				void TransposeBMatrixMultiplyTest();
		};
	}
}

#endif // OPENMM_LTMD_MATHTEST_H_
