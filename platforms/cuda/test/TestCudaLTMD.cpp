/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "OpenMM.h"
#include "openmm/serialization/XmlSerializer.h"
#include "../src/CudaKernelsNML.h"
#include "../../../test/AssertionUtilities.h"
#include "nmlopenmm/NMLIntegrator.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <openmm/State.h>
#include "nmlopenmm/LTMDParameters.h"

using namespace OpenMM;
using namespace OpenMM_LTMD;
using namespace std;

const double TOL = 1e-5;

void testMinimizationAndIntegration() {
    // Load the system.

    ifstream in("villin.xml");
    System* system = XmlSerializer::deserialize<System>(in);
    in.close();
    in.open("villin_minimize.txt");
    int numParticles = system->getNumParticles();

    // Load the starting positions.

    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        in >> positions[i][0];
        in >> positions[i][1];
        in >> positions[i][2];
    }
    in.close();

        int res[] = {21, 11, 12, 15, 12, 20, 16, 6, 10, 16,
                 20, 7, 17, 14, 13, 3, 3, 5, 11, 10,
		 20, 10, 9, 5, 19, 14, 19, 10, 14, 16,
		 6, 12, 5, 12, 5, 8, 3, 6, 19, 16,
		 6, 16, 6, 15, 16, 6, 7, 19};
    LTMDParameters ltmd;
    ltmd.delta = 1e-9;
    ltmd.bdof = 12;
    ltmd.res_per_block = 1;
    ltmd.modes = 20;
    for (int i = 0; i < 49; i++)
       ltmd.residue_sizes.push_back(res[i]);

    ltmd.forces.push_back(LTMDForce("CenterOfMass", 0));
    ltmd.forces.push_back(LTMDForce("Bond", 1));
    ltmd.forces.push_back(LTMDForce("Angle", 2));
    ltmd.forces.push_back(LTMDForce("Dihedral", 3));
    ltmd.forces.push_back(LTMDForce("Improper", 4));
    ltmd.forces.push_back(LTMDForce("Nonbonded", 5));


    // Create the integrator and context, then minimize it.

    int numModes = 10;
    NMLIntegrator integ(300, 100.0, 0.05, &ltmd);
    integ.setMaxEigenvalue(5e5);
    Context context(*system, integ, Platform::getPlatformByName("Reference"));
    context.setPositions(positions);
    double energy1 = context.getState(State::Energy).getPotentialEnergy();
    integ.minimize(50);

    // Verify that the energy decreased, and the slow modes were not modified during minimization.

    State state = context.getState(State::Positions | State::Energy);
    ASSERT(state.getPotentialEnergy() < energy1);
    vector<vector<Vec3> > modes = integ.getProjectionVectors();
    const vector<Vec3>& newPositions = state.getPositions();
    for (int i = 0; i < numModes; i++) {
        double oldValue = 0.0, newValue = 0.0;
        for (int j = 0; j < numParticles; j++) {
            double scale = sqrt(system->getParticleMass(j));
            oldValue += positions[j].dot(modes[i][j])*scale;
            newValue += newPositions[j].dot(modes[i][j])*scale;
        }
        ASSERT_EQUAL_TOL(oldValue, newValue, 1e-3);
    }

    // Simulate the system and see if the temperature is correct.

    integ.step(1000);
    integ.setFriction(10);
    double ke = 0.0;
    const int steps = 5000;
    for (int i = 0; i < steps; ++i) {
        State state = context.getState(State::Energy);
        ke += state.getKineticEnergy();
        integ.step(1);
    }
    ke /= steps;
    double expected = 0.5*numModes*BOLTZ*300;
    ASSERT_USUALLY_EQUAL_TOL(expected, ke, 6/std::sqrt((double) steps));

    delete system;
}

int main() {
    try {
        Platform::loadPluginsFromDirectory(Platform::getDefaultPluginsDirectory());
        testMinimizationAndIntegration();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
