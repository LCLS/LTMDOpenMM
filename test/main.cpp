#include <cppunit/XmlOutputter.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/BriefTestProgressListener.h>

int main( int argc, char *argv[] ) {
	// informs test-listener about testresults
	CppUnit::TestResult testresult;

	// register listener for collecting the test-results
	CppUnit::TestResultCollector collectedresults;
	testresult.addListener( &collectedresults );

	// insert test-suite at test-runner by registry
	CppUnit::TestRunner testrunner;
	testrunner.addTest( CppUnit::TestFactoryRegistry :: getRegistry().makeTest() );
	testrunner.run( testresult );

	// output results in compiler-format
	CppUnit::CompilerOutputter compileroutputter( &collectedresults, std::cerr );
	compileroutputter.write();

	// writing result on a XML file
	std::ofstream xmlFileOut( "testresults.xml" );
	CppUnit::XmlOutputter xmlOut( &collectedresults, xmlFileOut );
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}
