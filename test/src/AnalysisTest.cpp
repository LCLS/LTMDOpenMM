#include "AnalysisTest.h"

#include "LTMD/Analysis.h"

#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( LTMD::Analysis::Test );

namespace LTMD {
	namespace Analysis {
		void Test::BlockDiagonalize() {
            OpenMM::LTMD::Block block;
            
            // Read Block Data
            std::cout << "Reading Block" << std::endl;
            
            std::ifstream sBlock("data/block.txt");
            if( sBlock.good() ){
                unsigned int width, height;
                sBlock >> width >> height;
                
                std::cout << "\tWidth: " << width << " Height: " << height << std::endl;
                
                TNT::Array2D<double> data( width, height, 0.0 );
                
                for( int i = 0; i < width; i++ ){
                    for( int j = 0; j < height; j++ ){
                        sBlock >> data[i][j];
                    }
                }
                
                block.Data = data;
                block.StartAtom = 0;
                block.EndAtom = width/3;
            }
            sBlock.close();
            
            // Read Positions
            std::vector<OpenMM::Vec3> position;
            
            std::cout << "Reading Positions" << std::endl;
            
            std::ifstream sPos("data/block_positions.txt");
            if( sPos.good() ){
                unsigned int size;
                sPos >> size;
                
                std::cout << "\tCount:" << size << std::endl;
                
                for( int i = 0; i < size; i++ ){
                    double x, y, z;
                    sPos >> x >> y >> z;
                    
                    position.push_back( OpenMM::Vec3( x, y, z ) );
                }
            }
            sPos.close();
            
            // Read Masses
            std::cout << "Reading Masses" << std::endl;
            
            std::vector<double> mass;
            
            std::ifstream sMass("data/block_masses.txt");
            if( sMass.good() ){
                unsigned int size;
                sMass >> size;
                
                std::cout << "\tCount:" << size << std::endl;
                
                mass.resize( size );
                for( int i = 0; i < size; i++ ){
                    sMass >> mass[i];
                }
            }
            sMass.close();
            
            // Perform Calculation
            TNT::Array1D<double> eval( block.Data.dim1(), 0.0 );
			TNT::Array2D<double> evec( block.Data.dim1(), block.Data.dim1(), 0.0 );
              
            OpenMM::LTMD::Analysis::DiagonalizeBlock( block, position, mass, eval, evec );
            
            // Saving Vectors
            std::ofstream sVectors("vectors.txt");
            if( sVectors.good() ){
                sVectors.precision(10);
                
                for( int i = 0; i < evec.dim1(); i++ ){
                    for( int j = 0; j < evec.dim1(); j++ ){
                        sVectors << evec[i][j] << " ";
                    }
                    sVectors << std::endl;
                }
                
                sVectors.close();
            }
            
            // Read Expected Eigenvalues
            std::cout << "Reading Values" << std::endl;
            
            std::vector<double> expected_eval;
            
            std::ifstream sEval("data/block_values.txt");
            if( sEval.good() ){
                unsigned int size;
                sEval >> size;
                
                std::cout << "\tCount:" << size << std::endl;
                
                expected_eval.resize( size );
                for( int i = 0; i < size; i++ ){
                    sEval >> expected_eval[i];
                }
            }
            sEval.close();
            
            // Read Expected Eigenvectors
            std::vector<std::vector<double> > expected_evec;
            
            std::cout << "Reading Vectors" << std::endl;
            
            std::ifstream sEvec("data/block_vectors.txt");
            if( sEvec.good() ){
                unsigned int columns, rows;
                sEvec >> columns >> rows;
                
                std::cout << "\tColumns: " << columns << " Rows: " << rows << std::endl;
                
                expected_evec.resize( rows );
                for( int i = 0; i < rows; i++ ){
                    expected_evec[i].resize( columns );
                    for( int j = 0; j < columns; j++ ){
                        sEvec >> expected_evec[i][j];
                    }
                }
            }
            sEvec.close();
            
            // Compare Values
            std::cout << "Testing Values" << std::endl;
            int max_pos = 0;
            double max_diff = 0.0;
            for( int i = 0; i < eval.dim1(); i++ ){
                double diff = expected_eval[i] - eval[i];
                if( diff > max_diff ) {
                    max_pos = i;
                    max_diff = diff;
                }
                
                CPPUNIT_ASSERT_DOUBLES_EQUAL( expected_eval[i], eval[i], 1e-3 );
            }
            
            std::cout << "Max Error: " << max_diff << " : " << expected_eval[max_pos] << " " << eval[max_pos] << std::endl;
            for( int i = 0; i < eval.dim1(); i++ ){
                CPPUNIT_ASSERT_DOUBLES_EQUAL( expected_eval[i], eval[i], 1e-3 );
            }
            
            
            // Compare Vectors
            std::cout << "Testing Vectors" << std::endl;
            max_diff = 0.0;
            max_pos = 0;
            int max_pos_y = 0;
            for( int i = 0; i < evec.dim1(); i++ ){
                for( int j = 0; j < evec.dim2(); j++ ){
                    double diff = std::abs(expected_evec[i][j] - evec[i][j]);
                    
                    if( diff > max_diff ) {
                        max_pos = i;
                        max_pos_y = j;
                        max_diff = diff;
                    }
                }
            }
            
            std::cout << "Max IJ: " << max_pos << " " << max_pos_y << std::endl;
            std::cout << "Max Error: " << max_diff << " : " << expected_evec[max_pos][max_pos_y] << " " << evec[max_pos][max_pos_y] << std::endl;
            
            // Rewrite for Overlap
            for( int i = 0; i < evec.dim1(); i++ ){
                for( int j = 0; j < evec.dim2(); j++ ){
                    CPPUNIT_ASSERT_DOUBLES_EQUAL( std::abs(expected_evec[i][j]), std::abs(evec[i][j]), 1e-3 );
                }
            }
		}
	}
}
