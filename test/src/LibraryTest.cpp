#include "LibraryTest.h"

#include "LTMD/Analysis.h"
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( LTMD::Library::Test );

namespace LTMD {
	namespace Library {
		using namespace OpenMM::LTMD;
		
		void Test::SortEigenvalue() {
			std::vector<double> input;
			std::ifstream inStream( "data/UnsortedEigenvalues.txt" );
			while( !inStream.eof() ){
				double first = 0.0;
				inStream >> first;
				input.push_back( first );
			}
			input.pop_back();
			
			TNT::Array1D<double> data( input.size(), 0.0 );
			for( unsigned int i = 0; i < input.size(); i++ ){
				data[i] = input[i];
			}
			
			std::vector<EigenvalueColumn> expected;
			std::ifstream exStream( "data/SortedEigenvalues.txt" );
			while( !exStream.eof() ){
				int second = 0;
				double first = 0.0;
				exStream >> first >> second;
				expected.push_back( std::make_pair( first, second ) );
			}
			expected.pop_back();
			
			std::vector<EigenvalueColumn> output = Analysis::SortEigenvalues( data );
			
			CPPUNIT_ASSERT_EQUAL_MESSAGE( "Array Size Differs", expected.size(), output.size() );
			
			for( unsigned int i = 0; i < output.size(); i++ ) {
				std::ostringstream stream;
				stream << "Element " << i << " Differs: ";
				stream << expected[i].first << " " << output[i].first << " ";
				stream << expected[i].second << " " << output[i].second << " ";
			
				CPPUNIT_ASSERT_EQUAL_MESSAGE( stream.str(), expected[i].second, output[i].second );
				CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE( stream.str(), expected[i].first, output[i].first, 1e-3 );
			}
		}
		
		typedef std::vector<double> Row;
		typedef std::vector<Row> Matrix;
		
		typedef TNT::Array2D<double> TNTMatrix;
		
		const Matrix ReadMatrix( const std::string& path ){
			Matrix retVal;
			
			std::string line;
			std::ifstream matrix( path.c_str() );
			
			while( std::getline( matrix, line ) ){
				// Read Row
				Row row;
				
				std::istringstream stream( line );
				while( !stream.eof() ){
					double value = 0.0;
					stream >> value;
					
					row.push_back( value );
				}
				row.pop_back();
				
				// Append Row
				retVal.push_back( row );
			}
			
			return retVal;
		}
		
		const TNTMatrix ConvertMatrix( const Matrix& matrix ) {
			TNTMatrix retVal( matrix.size(), matrix[0].size() );
			
			for( unsigned int i = 0; i < matrix.size(); i++ ){
				for( unsigned int j = 0; j < matrix[i].size(); j++ ){
					retVal[i][j] = matrix[i][j];
				}
			}
			
			return retVal;
		}
		
		void Test::CalculateQ() {
			TNTMatrix E = ConvertMatrix( ReadMatrix( "data/EMatrix.txt" ) );
			TNTMatrix Q = ConvertMatrix( ReadMatrix( "data/QMatrix.txt" ) );
			TNTMatrix expected = ConvertMatrix( ReadMatrix( "data/UMatrix.txt" ) );
			
			Analysis analysis;
			TNTMatrix result = analysis.CalculateU(E, Q);
			
			CPPUNIT_ASSERT_EQUAL_MESSAGE( "Matrix I Dimension Differs", expected.dim1(), result.dim1() );
			CPPUNIT_ASSERT_EQUAL_MESSAGE( "Matrix J Dimension Differs", expected.dim2(), result.dim2() );
			
			for( unsigned int i = 0; i < expected.dim1(); i++ ){
				for( unsigned int j = 0; j < expected.dim2(); j++ ){
					std::ostringstream stream;
					stream << "Element " << i << " " << j << " Differs: ";
					
					CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE( stream.str(), expected[i][j], result[i][j], 1e-3 );
				}
			}	
		}
	}
}
