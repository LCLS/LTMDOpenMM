#include "OpenMM.h"
#include <vector>
#include <string>
#include <iostream>

using namespace OpenMM;
using namespace std;

// TEST subroutine testDynamicLoadingOfIntegrateNMLStepKernel()
//
// Tests basic mechanism of registering NML plugin, and verifies that
// the IntegrationNMLStep kernel is registered with the OpenMM Reference 
// platform.
// In order to pass this test, environment varibble OPENMM_PLUGIN_DIR
// must be set to a directory containing the NormalModeLangevin plugin.
// CMake version 2.8 or higher might be required to get that variable set properly
// in nightly builds.
void testDynamicLoadingOfIntegrateNMLStepKernel() 
{
    const string& pluginDir = Platform::getDefaultPluginsDirectory();
    cout << "Default plugins directory = " << pluginDir << endl;
    Platform::loadPluginsFromDirectory(pluginDir);

    int platformCount = Platform::getNumPlatforms();
    for (int p = 0; p < platformCount; ++p) {
        const Platform& platform = Platform::getPlatform(p);
        cout << platform.getName() << endl;
    }

    // Create a context, just to initialize all plugins
    System system;
    VerletIntegrator integrator(0.01);
    Context context(system, integrator);

    // Was NormalModeLangevin plugin loaded?
    vector<string> kernelName;
    kernelName.push_back("IntegrateNMLStep");
    cout << "Searching for kernel IntegrateNMLStep" << endl;
    Platform& platform = Platform::findPlatform(kernelName); // throws if no platform with kernel
    cout << "IntegrateNMLStep kernel found in " << platform.getName() << " platform." << endl;
}

int main(int argc, const char* argv[]) 
{
    try 
    {
        // testDynamicLoadingOfIntegrateNMLStepKernel may fail if OPENMM_PLUGIN_DIR is not set correctly
        testDynamicLoadingOfIntegrateNMLStepKernel();

        cout << "tests passed" << endl;
        return 0;
    } 
    catch (const std::exception& exc) 
    {
        cout << "FAILED: " << exc.what() << endl;
        return 1;
    }
}

