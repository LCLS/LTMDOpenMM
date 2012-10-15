#include "MathTest.h"

#include "LTMD/Math.h"

#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( LTMD::Math::Test );

namespace LTMD {
	namespace Math {
		void Test::EigenvalueTest() {
			Matrix a( 5, 5 ), vectors( 5, 5 );
            
			a(0,0) =  1.96; a(1,0) =  0.00; a(2,0) =  0.00; a(3,0) =  0.00; a(4,0) =  0.00;
			a(0,1) = -6.49; a(1,1) =  3.80; a(2,1) =  0.00; a(3,1) =  0.00; a(4,1) =  0.00;
			a(0,2) = -0.47; a(1,2) = -6.39; a(2,2) =  4.17; a(3,2) =  0.00; a(4,2) =  0.00;
            a(0,3) = -7.20; a(1,3) =  1.50; a(2,3) = -1.51; a(3,3) =  5.70; a(4,3) =  0.00;
            a(0,4) = -0.65; a(1,4) = -6.34; a(2,4) =  2.67; a(3,4) =  1.80; a(4,4) = -7.10;

            std::vector<double> values( 5 );
			FindEigenvalues( a, values, vectors );

            std::vector<double> expectedValues( 5 );
			expectedValues[0] = -11.0656;
			expectedValues[1] = -06.2287;
			expectedValues[2] =  00.8640;
			expectedValues[3] =  08.8655;
			expectedValues[4] =  16.0948;
            
			for( unsigned int i = 0; i < values.size(); i++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( expectedValues[i], values[i], 1e-2 );
			}
		}
        
        void Test::EigenvectorTest() {
			Matrix a( 5, 5 ), vectors( 5, 5 );
            
			a(0,0) =  1.96; a(1,0) =  0.00; a(2,0) =  0.00; a(3,0) =  0.00; a(4,0) =  0.00;
			a(0,1) = -6.49; a(1,1) =  3.80; a(2,1) =  0.00; a(3,1) =  0.00; a(4,1) =  0.00;
			a(0,2) = -0.47; a(1,2) = -6.39; a(2,2) =  4.17; a(3,2) =  0.00; a(4,2) =  0.00;
            a(0,3) = -7.20; a(1,3) =  1.50; a(2,3) = -1.51; a(3,3) =  5.70; a(4,3) =  0.00;
            a(0,4) = -0.65; a(1,4) = -6.34; a(2,4) =  2.67; a(3,4) =  1.80; a(4,4) = -7.10;
            
            std::vector<double> values( 5 );
			FindEigenvalues( a, values, vectors );
            
            Matrix expected( 5, 5 );

            expected(0,0) = -0.2981; expected(0,1) = -0.6075; expected(0,2) =  0.4026; expected(0,3) = -0.3745; expected(0,4) =  0.4896;
            expected(1,0) = -0.5078; expected(1,1) = -0.2880; expected(1,2) = -0.4066; expected(1,3) = -0.3572; expected(1,4) = -0.6053;
            expected(2,0) = -0.0816; expected(2,1) = -0.3843; expected(2,2) = -0.6600; expected(2,3) =  0.5008; expected(2,4) =  0.3991;
            expected(3,0) = -0.0036; expected(3,1) = -0.4467; expected(3,2) =  0.4553; expected(3,3) =  0.6204; expected(3,4) = -0.4564;
            expected(4,0) = -0.8041; expected(4,1) =  0.4480; expected(4,2) =  0.1725; expected(4,3) =  0.3108; expected(4,4) =  0.1622;
            
			std::vector<double> overlap( 5, 0.0f );
            
            for( size_t i = 0; i < vectors.Rows; i++ ) {
				double sum = 0.0;
				for( size_t j = 0; j < vectors.Rows; j++ ) {
					for( size_t k = 0; k < vectors.Columns; k++ ) {
						double dot = vectors(k,i) * expected(k,j);
						sum += ( dot * dot );
					}
				}
				overlap[i] = sum;
			}
            
			for( unsigned int i = 0; i < overlap.size(); i++ ) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0f, overlap[i], 1e-2 );
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
            
			for( unsigned int i = 0; i < c.Rows; i++ ) {
				for( unsigned int j = 0; j < c.Columns; j++ ) {
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
            
            for( unsigned int i = 0; i < c.Rows; i++ ) {
				for( unsigned int j = 0; j < c.Columns; j++ ) {
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
            
			for( unsigned int i = 0; i < c.Rows; i++ ) {
				for( unsigned int j = 0; j < c.Columns; j++ ) {
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
            
            for( unsigned int i = 0; i < c.Rows; i++ ) {
				for( unsigned int j = 0; j < c.Columns; j++ ) {
					CPPUNIT_ASSERT_DOUBLES_EQUAL( expected(i, j), c(i, j), 1e-3 );
				}
			}
		}
	}
}
