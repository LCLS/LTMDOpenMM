#include "MathTest.h"

#include "LTMD/Math.h"

#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( LTMD::Math::Test );

namespace LTMD {
	namespace Math {
		void Test::Eigenvalues() {
			Matrix a( 3, 3 ), vectors( 3, 3 );

			a(0,0) = 1.0; a(0,1) = 1.0; a(0,2) = 0.0;
			a(1,0) = 1.0; a(1,1) = 0.0; a(1,2) = 1.0;
			a(2,0) = 0.0; a(2,1) = 1.0; a(2,2) = 1.0;

            std::vector<double> values( 3 );

			FindEigenvalues( a, values, vectors );

            std::vector<double> expectedValues( 3 );
			expectedValues[0] = -1.0;
			expectedValues[1] =  1.0;
			expectedValues[2] =  2.0;

			for( unsigned int i = 0; i < values.size(); i++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( expectedValues[i], values[i], 1e-2 );
			}

			Matrix expectedVectors( 3, 3 );

			expectedVectors(0, 0) = -0.408; expectedVectors(0, 1) = -0.707; expectedVectors(0, 2) = -0.577;
			expectedVectors(1, 0) =  0.816; expectedVectors(1, 1) = -0.000; expectedVectors(1, 2) = -0.577;
			expectedVectors(2, 0) = -0.408; expectedVectors(2, 1) = 0.707; expectedVectors(2, 2) = -0.577;

			std::vector<double> overlap( 3 );

			for( unsigned int i = 0; i < vectors.Width; i++ ) {
				double sum = 0.0;
				for( unsigned int j = 0; j < vectors.Width; j++ ) {
					for( unsigned int k = 0; k < vectors.Height; k++ ) {
						double dot = vectors(k,i) * expectedVectors(k, j);
						sum += ( dot * dot );
					}
				}
				overlap[i] = sum;
			}

			for( unsigned int i = 0; i < values.size(); i++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, overlap[i], 1e-2 );
			}
		}

		void Test::MatrixMultiplyTest() {
			Matrix a( 3, 3 ), b( 3, 3 ), c( 3, 3 ), expected( 3, 3 );

			a(0, 0) = 0.0; a(0, 1) = 1.0; a(0, 2) = 2.0;
			a(1, 0) = 3.0; a(1, 1) = 4.0; a(1, 2) = 5.0;
			a(2, 0) = 6.0; a(2, 1) = 7.0; a(2, 2) = 8.0;

			b(0, 0) =  9.0; b(0, 1) = 10.0; b(0, 2) = 11.0;
			b(1, 0) = 12.0; b(1, 1) = 13.0; b(1, 2) = 14.0;
			b(2, 0) = 15.0; b(2, 1) = 16.0; b(2, 2) = 17.0;

			MatrixMultiply( a, b, c );

			expected(0, 0) =  42.0; expected(0, 1) =  45.0; expected(0, 2) =  48.0;
			expected(1, 0) = 150.0; expected(1, 1) = 162.0; expected(1, 2) = 174.0;
			expected(2, 0) = 258.0; expected(2, 1) = 279.0; expected(2, 2) = 300.0;

			for( unsigned int i = 0; i < c.Width; i++ ) {
				for( unsigned int j = 0; j < c.Height; j++ ) {
					CPPUNIT_ASSERT_DOUBLES_EQUAL( expected(i, j), c(i, j), 1e-3 );
				}
			}
		}
	}
}
