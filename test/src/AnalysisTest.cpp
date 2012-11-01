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
			/*std::cout << "Block Diagonalization" << std::endl;
			
			
			
			// Read Block Data
			std::ifstream sBlock("data/block.txt");
            unsigned int width, height;
            sBlock >> width >> height;
            
            OpenMM::LTMD::Block block( width, height );
            
            for( int i = 0; i < width; i++ ){
                for( int j = 0; j < height; j++ ){
                    sBlock >> block.Data(i,j);
                }
            }

            block.StartAtom = 0;
            block.EndAtom = width/3;
			
			// Read Positions
			std::vector<OpenMM::Vec3> position;

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
			std::vector<double> mass = Read1D("data/block_masses.txt");
			
			// Perform Calculation
            std::vector<double> eval( block.Data.Rows );
			Matrix evec( block.Data.Rows, block.Data.Columns );
			  
			OpenMM::LTMD::Analysis::DiagonalizeBlock( block, position, mass, eval, evec );
			
			// Read Expected Eigenvalues
			std::vector<double> expected_eval = Read1D("data/block_values.txt");
			
			// Read Expected Eigenvectors
			std::vector<std::vector<double> > expected_evec = Read2D("data/block_vectors.txt");
			
			// Compare Values
			int max_pos = 0;
			double max_diff = 0.0;
			for( int i = 0; i < eval.size(); i++ ){
				double diff = expected_eval[i] - eval[i];
				if( diff > max_diff ) {
					max_pos = i;
					max_diff = diff;
				}
			}
			
			// Compare Vectors
			max_diff = 0.0;
			max_pos = 0;
			int max_pos_y = 0;
			for( int i = 0; i < evec.Rows; i++ ){
				for( int j = 0; j < evec.Columns; j++ ){
					double diff = std::abs(expected_evec[i][j] - evec(i,j));
					
					if( diff > max_diff ) {
						max_pos = i;
						max_pos_y = j;
						max_diff = diff;
					}
				}
			}
			
			// Rewrite for Overlap
			for( int i = 0; i < eval.size(); i++ ){
				CPPUNIT_ASSERT_DOUBLES_EQUAL( expected_eval[i], eval[i], 1e-3 );
			}
			
			for( int i = 0; i < evec.Rows; i++ ){
				for( int j = 0; j < evec.Columns; j++ ){
					CPPUNIT_ASSERT_DOUBLES_EQUAL( std::abs(expected_evec[i][j]), std::abs(evec(i,j)), 1e-3 );
				}
			}*/
		}
		
		void Test::GeometricDOF(){
			std::cout << "Geometric DOF" << std::endl;
			
			// Read Eigenvalues
			std::vector<double> eval_data = Read1D("data/block_values.txt");
			
			// Read Eigenvectors
			std::vector<std::vector<double> > evec_data = Read2D("data/block_vectors.txt");
			
			// Read Block Data
			int start = 0, end = 0;
			
			std::ifstream sBlock("data/block.txt");
			if( sBlock.good() ){
				unsigned int width, height;
				sBlock >> width >> height;
				
				start = 0;
				end = width;
			}
			sBlock.close();
			
			// Read Positions
			std::vector<OpenMM::Vec3> position;
			
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
			std::vector<double> mass = Read1D("data/block_masses.txt");
			
			// Perform Calculation
			const unsigned int size = end;
            std::vector<double> eval( size, 0.0 );
			for( int i = 0; i < eval.size(); i++ ){
				eval[i] = eval_data[i];
			}
			
			Matrix evec( size, size );
			for( int i = 0; i < evec.Rows; i++ ){
				for( int j = 0; j < evec.Columns; j++ ){
					evec(i,j) = evec_data[i][j];
				}
			}
			
			OpenMM::LTMD::Analysis::GeometricDOF(size, start, end, position, mass, eval, evec);
			
			// Write Vectors
			std::ofstream sv("vectors.txt");
			if( sv.good() ){
				for( int i = 0; i < evec.Rows; i++ ){
					for( int j = 0; j < evec.Columns; j++ ){
						sv << evec(i,j) << " ";
					}
					sv << std::endl;
				}
			}
			sv.close();
			
			// Read Expected Eigenvalues
			std::vector<double> expected_eval = Read1D("data/block_gdof_values.txt");
			
			// Read Expected Eigenvectors
			std::vector<std::vector<double> > expected_evec = Read2D("data/block_gdof_vectors.txt");
			
			// Compare Values
			std::cout << "Testing Values" << std::endl;
			int max_pos = 0;
			double max_diff = 0.0;
			for( int i = 0; i < eval.size(); i++ ){
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
			int ierror, jerror;
			
			for( int j = 0; j < evec.Columns; j++ ){
				double max_j_diffp = 0.0, max_j_diffm = 0.0;
				int im = 0, ip = 0, jm = 0, jp = 0;
				
				for( int i = 0; i < evec.Rows; i++ ){
					double diff = std::abs(expected_evec[i][j] - evec(i,j));
					double diffp = std::abs(expected_evec[i][j] + evec(i,j));
					
					if( diff > max_j_diffm ) {
						im=i;
						jm=j;
						max_j_diffm = diff;
					}
					
					if( diffp > max_j_diffp ) {
						ip=i;
						jp=j;
						max_j_diffp = diffp;
					}
				}
				
				double temp_diff = max_j_diffm;
				int iout = im;
				int jout = jm;
				
				if( max_j_diffp < max_j_diffm){
					iout = ip;
					jout = jp;
					temp_diff = max_j_diffp;
				}
				
				if(temp_diff > max_diff){
					ierror = iout;
					jerror = jout;
					max_diff = temp_diff;
				}
			}
		
			std::cout << "IJ: " << ierror << ", " << jerror << std::endl;
			std::cout << "Max Error: " << max_diff << " : " << expected_evec[ierror][jerror] << ", " << evec(ierror,jerror) << std::endl;
			
			// Rewrite for Overlap
			for( int i = 0; i < eval.size(); i++ ){
				CPPUNIT_ASSERT_DOUBLES_EQUAL( expected_eval[i], eval[i], 1e-3 );
			}
			
			for( int i = 0; i < evec.Rows; i++ ){
				for( int j = 0; j < evec.Columns; j++ ){
					CPPUNIT_ASSERT_DOUBLES_EQUAL( std::abs(expected_evec[i][j]), std::abs(evec(i,j)), 1e-3 );
				}
			}
		}
	}
}
