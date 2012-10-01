#include "MathTest.h"

#include "LTMD/Math.h"

#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( LTMD::Math::Test );

namespace LTMD {
	namespace Math {
		void Test::EigenvalueTest() {
			Matrix a( 3, 3 ), vectors( 3, 3 );

			a(0,0) = 1.0; a(0,1) = 2.0; a(0,2) = 3.0;
			a(1,0) = 4.0; a(1,1) = 5.0; a(1,2) = 6.0;
			a(2,0) = 7.0; a(2,1) = 8.0; a(2,2) = 9.0;

            std::vector<double> values( 3 );
			FindEigenvalues( a, values, vectors );

            std::vector<double> expectedValues( 3 );
			expectedValues[0] = 16.1168;
			expectedValues[1] = -1.1168;
			expectedValues[2] = -0.0000;

			for( unsigned int i = 0; i < values.size(); i++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( expectedValues[i], values[i], 1e-2 );
			}
		}
        
        void Test::EigenvectorTest() {
			Matrix a( 3, 3 ), vectors( 3, 3 );
            
			a(0,0) = 1.0; a(0,1) = 2.0; a(0,2) = 3.0;
			a(1,0) = 4.0; a(1,1) = 5.0; a(1,2) = 6.0;
			a(2,0) = 7.0; a(2,1) = 8.0; a(2,2) = 9.0;
            
            std::vector<double> values( 3 );
			FindEigenvalues( a, values, vectors );
            
            Matrix expectedVectors( 3, 3 );
            
			expectedVectors(0, 0) = -0.2320; expectedVectors(0, 1) = -0.7858; expectedVectors(0, 2) =  0.4082;
			expectedVectors(1, 0) = -0.5253; expectedVectors(1, 1) = -0.0868; expectedVectors(1, 2) = -0.8165;
			expectedVectors(2, 0) = -0.8187; expectedVectors(2, 1) =  0.6123; expectedVectors(2, 2) =  0.4082;
            
			std::vector<double> overlap( 3, 0.0f );
            
            for( size_t i = 0; i < vectors.Width; i++ ) {
				double sum = 0.0;
				for( size_t j = 0; j < vectors.Width; j++ ) {
					for( size_t k = 0; k < vectors.Height; k++ ) {
						double dot = vectors(k,i) * expectedVectors(k,j);
						sum += ( dot * dot );
					}
				}
				overlap[i] = sum;
			}
            
			for( unsigned int i = 0; i < overlap.size(); i++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0, overlap[i], 1e-2 );
			}
		}

		void Test::MatrixMultiplyTest() {
			Matrix a( 2, 3 ), b( 3, 2 ), c( 2, 2 ), expected( 2, 2 );

            // [ 0.0, 1.0, 2.0; 3.0, 4.0, 5.0 ]
			a(0, 0) = 0.0; a(0, 1) = 1.0; a(0, 2) = 2.0;
			a(1, 0) = 3.0; a(1, 1) = 4.0; a(1, 2) = 5.0;

            // [ 9.0, 10.0; 12.0, 13.0; 15.0, 16.0 ]
			b(0, 0) =  9.0; b(0, 1) = 10.0;
			b(1, 0) = 12.0; b(1, 1) = 13.0;
			b(2, 0) = 15.0; b(2, 1) = 16.0;

			MatrixMultiply( a, false, b, false, c );
            
			expected(0, 0) =  42.0; expected(0, 1) =  45.0;
			expected(1, 0) = 150.0; expected(1, 1) = 162.0;
            
			for( unsigned int i = 0; i < c.Width; i++ ) {
				for( unsigned int j = 0; j < c.Height; j++ ) {
					CPPUNIT_ASSERT_DOUBLES_EQUAL( expected(i, j), c(i, j), 1e-3 );
				}
			}
		}
        
        void Test::TransposeMatrixMultiplyTest() {
			Matrix a( 3, 2 ), b( 2, 3 ), c( 2, 2 ), expected( 2, 2 );
            
            // [ 0.0, 1.0; 2.0, 3.0; 4.0, 5.0 ]
			a(0, 0) = 0.0; a(0, 1) = 1.0;
            a(1, 0) = 2.0; a(1, 1) = 3.0;
            a(2, 0) = 4.0; a(2, 1) = 5.0;
            
            // [ 9.0, 10.0, 12.0; 13.0, 15.0, 16.0 ]
			b(0, 0) =  9.0; b(0, 1) = 10.0; b(0, 2) = 12.0;
            b(1, 0) = 13.0; b(1, 1) = 15.0; b(1, 2) = 16.0;
            
			MatrixMultiply( a, true, b, true, c );
            
			expected(0, 0) = 68.0; expected(0, 1) =  94.0;
			expected(1, 0) = 99.0; expected(1, 1) = 138.0;
            
            for( unsigned int i = 0; i < c.Width; i++ ) {
				for( unsigned int j = 0; j < c.Height; j++ ) {
					CPPUNIT_ASSERT_DOUBLES_EQUAL( expected(i, j), c(i, j), 1e-3 );
				}
			}
		}
        
        void Test::TransposeAMatrixMultiplyTest() {
            Matrix a( 3, 2 ), b( 3, 2 ), c( 2, 2 ), expected( 2, 2 );
            
            // [ 0.0, 1.0; 2.0, 3.0; 4.0, 5.0 ]
			a(0, 0) = 0.0; a(0, 1) = 1.0;
            a(1, 0) = 2.0; a(1, 1) = 3.0;
            a(2, 0) = 4.0; a(2, 1) = 5.0;
            
            // [ 9.0, 10.0; 12.0, 13.0; 15.0, 16.0 ]
			b(0, 0) =  9.0; b(0, 1) = 10.0;
			b(1, 0) = 12.0; b(1, 1) = 13.0;
			b(2, 0) = 15.0; b(2, 1) = 16.0;
            
			MatrixMultiply( a, true, b, false, c );
            
			expected(0, 0) =  84.0; expected(0, 1) =  90.0;
			expected(1, 0) = 120.0; expected(1, 1) = 129.0;
            
			for( unsigned int i = 0; i < c.Width; i++ ) {
				for( unsigned int j = 0; j < c.Height; j++ ) {
					CPPUNIT_ASSERT_DOUBLES_EQUAL( expected(i, j), c(i, j), 1e-3 );
				}
			}

		}
        
        void Test::TransposeBMatrixMultiplyTest() {
			Matrix a( 2, 3 ), b( 2, 3 ), c( 2, 2 ), expected( 2, 2 );
            
            // [ 0.0, 1.0, 2.0; 3.0, 4.0, 5.0 ]
			a(0, 0) = 0.0; a(0, 1) = 1.0; a(0, 2) = 2.0;
			a(1, 0) = 3.0; a(1, 1) = 4.0; a(1, 2) = 5.0;
            
            // [ 9.0, 10.0, 12.0; 13.0, 15.0, 16.0 ]
			b(0, 0) =  9.0; b(0, 1) = 10.0; b(0, 2) = 12.0;
            b(1, 0) = 13.0; b(1, 1) = 15.0; b(1, 2) = 16.0;
            
			MatrixMultiply( a, false, b, true, c );
            
			expected(0, 0) =  34.0; expected(0, 1) =  47.0;
			expected(1, 0) = 127.0; expected(1, 1) = 179.0;
            
            for( unsigned int i = 0; i < c.Width; i++ ) {
				for( unsigned int j = 0; j < c.Height; j++ ) {
					CPPUNIT_ASSERT_DOUBLES_EQUAL( expected(i, j), c(i, j), 1e-3 );
				}
			}
		}
	}
}
