#include "AnalysisTest.h"

#include "LTMD/Analysis.h"

#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( LTMD::Analysis::Test );

namespace LTMD {
	namespace Analysis {
        const std::vector<double> Read1D( const std::string& path ){
            std::vector<double> retVal;
            
            std::ifstream stream( path.c_str() );
            if( stream.good() ){
                unsigned int size = 0;
                stream >> size;
                
                retVal.resize( size );
                for( int i = 0; i < size; i++ ){
                    stream >> retVal[i];
                }
            }
            stream.close();
            
            return retVal;
        }
        
        const std::vector<std::vector<double> > Read2D( const std::string& path ){
            std::vector<std::vector<double> > retVal;
            
            std::ifstream stream( path.c_str() );
            if( stream.good() ){
                unsigned int width = 0, height = 0;
                stream >> width >> height;
                
                retVal.resize( height );
                for( int i = 0; i < retVal.size(); i++ ){
                    retVal[i].resize(width);
                    for( int j = 0; j < retVal[i].size(); j++ ){
                        stream >> retVal[i][j];
                    }
                }
            }
            stream.close();
            
            return retVal;
        }
        
		void Test::BlockDiagonalize() {
            std::cout << std::endl << "Block Diagonalize" << std::endl;
            
            OpenMM::LTMD::Block block;
            
            // Read Block Data
            std::cout << "Reading Block" << std::endl;
            
            std::ifstream sBlock("data/block.txt");
            if( sBlock.good() ){
                unsigned int width, height;
                sBlock >> width >> height;
                
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
                
                for( int i = 0; i < size; i++ ){
                    double x, y, z;
                    sPos >> x >> y >> z;
                    
                    position.push_back( OpenMM::Vec3( x, y, z ) );
                }
            }
            sPos.close();
            
            // Read Masses
            std::cout << "Reading Masses" << std::endl;
            std::vector<double> mass = Read1D("data/block_masses.txt");
            
            // Perform Calculation
            TNT::Array1D<double> eval( block.Data.dim1(), 0.0 );
			TNT::Array2D<double> evec( block.Data.dim1(), block.Data.dim1(), 0.0 );
              
            OpenMM::LTMD::Analysis::DiagonalizeBlock( block, position, mass, eval, evec );
            
            // Read Expected Eigenvalues
            std::cout << "Reading Values" << std::endl;
            
            std::vector<double> expected_eval = Read1D("data/block_values.txt");
            
            // Read Expected Eigenvectors
            std::vector<std::vector<double> > expected_evec = Read2D("data/block_vectors.txt");
            
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
            }
            
            std::cout << "Max Error: " << max_diff << " : " << expected_eval[max_pos] << " " << eval[max_pos] << std::endl;
            
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
            for( int i = 0; i < eval.dim1(); i++ ){
                CPPUNIT_ASSERT_DOUBLES_EQUAL( expected_eval[i], eval[i], 1e-3 );
            }
            
            for( int i = 0; i < evec.dim1(); i++ ){
                for( int j = 0; j < evec.dim2(); j++ ){
                    CPPUNIT_ASSERT_DOUBLES_EQUAL( std::abs(expected_evec[i][j]), std::abs(evec[i][j]), 1e-3 );
                }
            }
            
            std::cout << std::endl;
		}
        
        void Test::GeometricDOF(){
            std::cout << std::endl << "Geometric DOF" << std::endl;
            
            // Read Eigenvalues
            std::cout << "Reading Values" << std::endl;
            std::vector<double> eval_data = Read1D("data/block_values.txt");
            
            // Read Eigenvectors
            std::cout << "Reading Vectors" << std::endl;
            std::vector<std::vector<double> > evec_data = Read2D("data/block_vectors.txt");
            
            // Read Block Data
            int start = 0, end = 0;
            std::cout << "Reading Block" << std::endl;
            
            std::ifstream sBlock("data/block.txt");
            if( sBlock.good() ){
                unsigned int width, height;
                sBlock >> width >> height;
                
                std::cout << "\tWidth: " << width << " Height: " << height << std::endl;
                
                start = 0;
                end = width;
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
            std::vector<double> mass = Read1D("data/block_masses.txt");
            
            // Perform Calculation
            const unsigned int size = end;
            TNT::Array1D<double> eval( size, 0.0 );
            for( int i = 0; i < eval.dim1(); i++ ){
                eval[i] = eval_data[i];
            }
            
			TNT::Array2D<double> evec( size, size, 0.0 );
            for( int i = 0; i < evec.dim1(); i++ ){
                for( int j = 0; j < evec.dim2(); j++ ){
                    evec[i][j] = evec_data[i][j];
                }
            }
            
            OpenMM::LTMD::Analysis::GeometricDOF(size, start, end, position, mass, eval, evec);
            
            // Write Vectors
            std::ofstream sv("vectors.txt");
            if( sv.good() ){
                for( int i = 0; i < evec.dim1(); i++ ){
                    for( int j = 0; j < evec.dim2(); j++ ){
                        sv << evec[i][j] << " ";
                    }
                    sv << std::endl;
                }
            }
            sv.close();
            
            // Read Expected Eigenvalues
            std::cout << "Reading Values" << std::endl;
            std::vector<double> expected_eval = Read1D("data/block_gdof_values.txt");
            
            // Read Expected Eigenvectors
            std::cout << "Reading Vectors" << std::endl;
            std::vector<std::vector<double> > expected_evec = Read2D("data/block_gdof_vectors.txt");
            
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
            }
            
            std::cout << "Max Error: " << max_diff << " : " << expected_eval[max_pos] << " " << eval[max_pos] << std::endl;
            
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
            for( int i = 0; i < eval.dim1(); i++ ){
                CPPUNIT_ASSERT_DOUBLES_EQUAL( expected_eval[i], eval[i], 1e-3 );
            }
            
            for( int i = 0; i < evec.dim1(); i++ ){
                for( int j = 0; j < evec.dim2(); j++ ){
                    CPPUNIT_ASSERT_DOUBLES_EQUAL( std::abs(expected_evec[i][j]), std::abs(evec[i][j]), 1e-3 );
                }
            }
            
            std::cout << std::endl;
        }
	}
}
