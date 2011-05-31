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
#include "../include/ReferenceNMLDynamics.h"
#include "../../../test/AssertionUtilities.h"
#include "nmlopenmm/NMLIntegrator.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <openmm/State.h>

using namespace OpenMM;
using namespace OpenMM_LTMD;
using namespace std;

const double TOL = 1e-5;

void testProjection() {
    // Load the file containing data to project.

    ifstream in("projectedForces.txt");
    int numFrames;
    in >> numFrames;
    int numAtoms;
    in >> numAtoms;
    vector<RealOpenMM> mass(numAtoms);
    vector<RealOpenMM> invMass(numAtoms);
    vector<RealVec> initial(numAtoms);
    vector<RealVec> expected(numAtoms);
    vector<RealVec> projected(numAtoms);
    string type;
    for (int i = 0; i < numAtoms; i++) {
        in >> type;
        in >> initial[i][0];
        in >> initial[i][1];
        in >> initial[i][2];
        if (type[0] == 'H')
            mass[i] = 1.007;
        else if (type[0] == 'C')
            mass[i] = 12.0;
        else if (type[0] == 'N')
            mass[i] = 14.003;
        else if (type[0] == 'O')
            mass[i] = 15.994;
        else if (type[0] == 'S')
            mass[i] = 31.972;
        else
            throw OpenMMException("Unknown atom type: "+type);
        invMass[i] = 1.0/mass[i];
    }
    string skip;
    in >> skip;
    for (int i = 0; i < numAtoms; i++) {
        in >> type;
        in >> expected[i][0];
        in >> expected[i][1];
        in >> expected[i][2];
    }
    in.close();

    in.open("output.txt");
    in >> skip;
    for (int i = 0; i < numAtoms; i++) {
        in >> initial[i][0];
        in >> initial[i][1];
        in >> initial[i][2];
    }
    in >> skip;
    for (int i = 0; i < numAtoms; i++) {
        in >> expected[i][0];
        in >> expected[i][1];
        in >> expected[i][2];
    }
    in.close();

    // Load the normal modes.

    int numModes = 15;
    RealOpenMM* modes = new RealOpenMM[numModes*3*numAtoms];
    in.open("ww-fip35evec.txt");
    int index = 0;
    for (int i = 0; i < numModes; i++)
        for (int j = 0; j < numAtoms; j++) {
            int atom;;
            in >> atom;
            in >> modes[index++];
            in >> modes[index++];
            in >> modes[index++];
        }

    // Perform a projection and see if the results are correct.

    ReferenceNMLDynamics dynamics(numAtoms, 0.001, 1, 300, modes, numModes, 0, 1);
    dynamics.subspaceProjection(initial, projected, numAtoms, invMass, mass, false);
    for (int i = 0; i < numAtoms; i++) {
        ASSERT_EQUAL_TOL(expected[i][0], projected[i][0], 1e-2);
        ASSERT_EQUAL_TOL(expected[i][1], projected[i][1], 1e-2);
        ASSERT_EQUAL_TOL(expected[i][2], projected[i][2], 1e-2);
    }
}

void testMinimizationAndIntegration() {
    // Load the system.

    ifstream in("villin.xml");
    System* system = XmlSerializer::deserialize<System>(in);
    in.close();
    in.open("villin_start.txt");
    int numParticles = system->getNumParticles();

    // Load the starting positions.

    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        in >> positions[i][0];
        in >> positions[i][1];
        in >> positions[i][2];
    }
    in.close();

    // Create the integrator and context, then minimize it.

    int numModes = 10;
    NMLIntegrator integ(300, 100.0, 0.05);
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
        testProjection();
        testMinimizationAndIntegration();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
