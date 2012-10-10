#include "MathTest.h"

#include "LTMD/Math.h"

#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( LTMD::Math::Test );

namespace LTMD {
	namespace Math {
		void Test::EigenvalueTest() {
			Matrix a( 5, 5 ), vectors( 5, 5 );
            
			a(0,0) =  1.96; a(1,0) = 0.00; a(2,0) = 0.00; a(3,0) = 0.00; a(4,0) = 0.00;
			a(0,1) = -6.49; a(1,1) = 3.80; a(2,1) = 0.00; a(3,1) = 0.00; a(4,1) = 0.00;
			a(0,2) = -0.47; a(1,2) = -6.39; a(2,2) = 4.17; a(3,2) = 0.00; a(4,2) = 0.00;
            a(0,3) = -7.20; a(1,3) = 1.50; a(2,3) = -1.51; a(3,3) = 5.70; a(4,3) = 0.00;
            a(0,4) = -0.65; a(1,4) = -6.34; a(2,4) = 2.67; a(3,4) = 1.80; a(4,4) = -7.10;

            std::vector<double> values( 5 );
			FindEigenvalues( a, values, vectors );

            std::vector<double> expectedValues( 5 );
			expectedValues[0] = -11.07;
			expectedValues[1] = -6.23;
			expectedValues[2] = 0.86;
            expectedValues[3] = 8.87;
			expectedValues[4] = 16.09;
            
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
            
            vectors.Print();
            
            Matrix expected( 5, 5 );
            
            expected(0,0) = -0.30; expected(0,1) = -0.61; expected(0,2) =  0.40; expected(0,3) = -0.37; expected(0,4) =  0.49;
			expected(1,0) = -0.51; expected(1,1) = -0.29; expected(1,2) = -0.41; expected(1,3) = -0.36; expected(1,4) = -0.61;
			expected(2,0) = -0.08; expected(2,1) = -0.38; expected(2,2) = -0.66; expected(2,3) =  0.50; expected(2,4) =  0.40;
            expected(3,0) =  0.00; expected(3,1) = -0.45; expected(3,2) =  0.46; expected(3,3) =  0.62; expected(3,4) = -0.46;
            expected(4,0) = -0.80; expected(4,1) = -6.34; expected(4,2) =  0.17; expected(4,3) =  0.31; expected(4,4) =  0.16;
            
            std::cout << std::endl;
            expected.Print();
            
			std::vector<double> overlap( 5, 0.0f );
            
            for( size_t i = 0; i < vectors.Rows; i++ ) {
                std::cout << i << std::endl;
                
				double sum = 0.0;
				for( size_t j = 0; j < vectors.Rows; j++ ) {
					for( size_t k = 0; k < vectors.Columns; k++ ) {
						double dot = vectors(k,i) * expected(k,j);
						sum += ( dot * dot );
					}
				}
				overlap[i] = sum;
                
                std::cout << sum << std::endl;
			}
            
			for( unsigned int i = 0; i < overlap.size(); i++ ) {
                std::cout << overlap[i] << std::endl;
                //CPPUNIT_ASSERT_DOUBLES_EQUAL( 1.0f, overlap[i], 1e-2 );
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
