/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009 Stanford University and the Authors.           *
 * Authors: Chris Sweet                                                       *
 * Contributors: Christopher Bruns                                            *
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

#include "nmlopenmm/NMLIntegrator.h"
#include "nmlopenmm/IntegrateNMLStepKernel.h"
#include "nmlopenmm/NormalModeAnalysis.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"
#include <ctime>
#include <string>

using namespace OpenMM;
using namespace OpenMM_LTMD;
using std::string;
using std::vector;

NMLIntegrator::NMLIntegrator(double temperature, double frictionCoeff, double stepSize) : stepsSinceDiagonalize(0), rediagonalizeFrequency(1000) {
    setTemperature(temperature);
    setFriction(frictionCoeff);
    setStepSize(stepSize);
    setConstraintTolerance(1e-4);
    setMinimumLimit(0.1);
    setRandomNumberSeed((int) time(NULL));
}

void NMLIntegrator::initialize(ContextImpl& contextRef) {
    context = &contextRef;
    if (context->getSystem().getNumConstraints() > 0)
        throw OpenMMException("NMLIntegrator does not support systems with constraints");
    kernel = context->getPlatform().createKernel(IntegrateNMLStepKernel::Name(), contextRef);
    dynamic_cast<IntegrateNMLStepKernel&>(kernel.getImpl()).initialize(contextRef.getSystem(), *this);
}

vector<string> NMLIntegrator::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(IntegrateNMLStepKernel::Name());
    return names;
}

void NMLIntegrator::step(int steps) {
    for (int i = 0; i < steps; ++i) {
        context->updateContextState();

        if (eigenvectors.size() == 0 || stepsSinceDiagonalize%rediagonalizeFrequency == 0)
            computeProjectionVectors();
        stepsSinceDiagonalize++;

        context->calcForcesAndEnergy(true, false);

        // Integrate one step
        dynamic_cast<IntegrateNMLStepKernel&>(kernel.getImpl()).execute(*context, *this, 0.0, 1);
        //in case projection vectors changed, clear flag
        eigVecChanged = false;

        //minimize compliment space, set maximum minimizer loops to 50
        minimize(50);

        // Update the time and step counter.
        dynamic_cast<IntegrateNMLStepKernel&>(kernel.getImpl()).execute(*context, *this, 0.0, 2);
    }

    //update time
    context->setTime(context->getTime()+getStepSize() * steps);

}

void NMLIntegrator::minimize(int maxsteps) {

    if (eigenvectors.size() == 0)
        computeProjectionVectors();
    
    //minimum limit
    const double minlim = getMinimumLimit();

    // Record initial positions.
    dynamic_cast<IntegrateNMLStepKernel&>(kernel.getImpl()).execute(*context, *this, 0.0, 6);

    //loop
    double initialPE = context->calcForcesAndEnergy(true, true);
    for (int i = 0; i < maxsteps; ++i) {

        //minimize (try simple first)
        dynamic_cast<IntegrateNMLStepKernel&>(kernel.getImpl()).execute(*context, *this, initialPE, 3);    //stepType 3 is simple minimizer
        eigVecChanged = false;

        //minimize, uses quadratic soultion as 'stepsize' not forced
        const double currentPE = context->calcForcesAndEnergy(true, true);
        dynamic_cast<IntegrateNMLStepKernel&>(kernel.getImpl()).execute(*context, *this, currentPE, 4); //stepType 4 is quadratic minimizer

        //break if satisfies end condition
        const double quadraticPE = context->calcForcesAndEnergy(true, true);
        const double diff = initialPE - quadraticPE;
        if (diff < minlim && diff >= 0.0)
            break;

        // Accept or reject the step
        dynamic_cast<IntegrateNMLStepKernel&>(kernel.getImpl()).execute(*context, *this, currentPE, diff < 0.0 ? 5 : 6);
        if (diff < 0.0)
            context->calcForcesAndEnergy(true, false);
        else
            initialPE = quadraticPE;
    }

}

void NMLIntegrator::computeProjectionVectors() {
    NormalModeAnalysis nma;
    nma.computeEigenvectorsFull(*context, 20);
    const vector<vector<Vec3> > e1 = nma.getEigenvectors();
    setProjectionVectors(nma.getEigenvectors());
    maxEigenvalue = 5e5;//nma.getMaxEigenvalue();
    stepsSinceDiagonalize = 0;
}
