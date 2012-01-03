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
	}
}
