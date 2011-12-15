#include "MathTest.h"

#include "LTMD/Math.h"

#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( LTMD::Math::Test );

namespace LTMD {
	namespace Math {
		void Test::Eigenvalues() {
			TNT::Array2D<double> a( 3, 3, 0.0 ), vectors( 3, 3, 0.0 );

			a[0][0] = 1.0; a[0][1] = 1.0; a[0][2] = 0.0;
			a[1][0] = 1.0; a[1][1] = 0.0; a[1][2] = 1.0;
			a[2][0] = 0.0; a[2][1] = 1.0; a[2][2] = 1.0;
			
			TNT::Array1D<double> values( 3 );
			
			FindEigenvalues( a, values, vectors );
			
			TNT::Array1D<double> expectedValues( 3 );
			expectedValues[0] = -1.0;
			expectedValues[1] =  1.0;
			expectedValues[2] =  2.0;
			
			for( unsigned int i = 0; i < values.dim(); i++ ){
				CPPUNIT_ASSERT_DOUBLES_EQUAL( expectedValues[i], values[i], 1e-3);
			}
			
			TNT::Array2D<double> expectedVectors( 3, 3, 0.0 );
			
			expectedVectors[0][0] = -0.408; expectedVectors[0][1] =  0.707; expectedVectors[0][2] = -0.577;
			expectedVectors[1][0] =  0.816; expectedVectors[1][1] = -0.000; expectedVectors[1][2] = -0.577;
			expectedVectors[2][0] = -0.408; expectedVectors[2][1] = -0.707; expectedVectors[2][2] = -0.577;
			
			std::cout << "Result: " << std::endl;
			for( unsigned int i = 0; i < vectors.dim1(); i++ ){
				for( unsigned int j = 0; j < vectors.dim2(); j++ ){
					std::cout << vectors[i][j] << " ";
				}
				std::cout << std::endl;
			}
			
			for( unsigned int i = 0; i < vectors.dim1(); i++ ){
				for( unsigned int j = 0; j < vectors.dim2(); j++ ){
					CPPUNIT_ASSERT_DOUBLES_EQUAL( expectedVectors[i][j], vectors[i][j], 1e-3);
				}
			}
		}
		
		void Test::MatrixMultiplyTest() {
			TNT::Array2D<double> a( 3, 3, 0.0 ), b( 3, 3, 0.0 ), c( 3, 3, 0.0 ), expected( 3, 3, 0.0 );
			
			a[0][0] = 0.0; a[0][1] = 1.0; a[0][2] = 2.0;
			a[1][0] = 3.0; a[1][1] = 4.0; a[1][2] = 5.0;
			a[2][0] = 6.0; a[2][1] = 7.0; a[2][2] = 8.0;
			
			b[0][0] =  9.0; b[0][1] = 10.0; b[0][2] = 11.0;
			b[1][0] = 12.0; b[1][1] = 13.0; b[1][2] = 14.0;
			b[2][0] = 15.0; b[2][1] = 16.0; b[2][2] = 17.0;
						
			MatrixMultiply( a, b, c );
			
			expected[0][0] =  42.0; expected[0][1] =  45.0; expected[0][2] =  48.0;
			expected[1][0] = 150.0; expected[1][1] = 162.0; expected[1][2] = 174.0;
			expected[2][0] = 258.0; expected[2][1] = 279.0; expected[2][2] = 300.0;
			
			for( unsigned int i = 0; i < c.dim1(); i++ ){
				for( unsigned int j = 0; j < c.dim2(); j++ ){
					CPPUNIT_ASSERT_DOUBLES_EQUAL( expected[i][j], c[i][j], 1e-3);
				}
			}
		}
	}
}
