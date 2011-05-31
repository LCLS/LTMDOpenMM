#ifndef REFERENCE_INTEGRATE_NML_STEP_KERNEL_H
#define REFERENCE_INTEGRATE_NML_STEP_KERNEL_H


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

#include "ReferencePlatform.h"
#include "ReferenceNMLDynamics.h"
#include "nmlopenmm/IntegrateNMLStepKernel.h"

namespace OpenMM_LTMD {

/**
 * This kernel is invoked by NMLIntegrator to take one time step.
 */
class ReferenceIntegrateNMLStepKernel : public IntegrateNMLStepKernel {
    public:
        ReferenceIntegrateNMLStepKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ReferencePlatform::PlatformData& data) : IntegrateNMLStepKernel(name, platform),
        data(data), dynamics(0), constraints(0), masses(0), constraintDistances(0), constraintIndices(0), projectionVectors(0) {
        }
        ~ReferenceIntegrateNMLStepKernel();
        /**
         * Initialize the kernel, setting up the particle masses.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the NMLIntegrator this kernel will be used for
         */
        void initialize(const OpenMM::System& system, const NMLIntegrator& integrator);
        /**
         * Execute the kernel.
         *
         * @param context    the context in which to execute this kernel
         * @param integrator the NMLIntegrator this kernel is being used for
         */
        void execute(OpenMM::ContextImpl& context, const NMLIntegrator& integrator, const double currentPE, const int stepType);
    private:
        OpenMM::ReferencePlatform::PlatformData& data;
        ReferenceNMLDynamics* dynamics;
        ReferenceConstraintAlgorithm* constraints;
        std::vector<RealOpenMM> masses;
        RealOpenMM* constraintDistances;
        int** constraintIndices;
        int numConstraints;
        double prevTemp, prevFriction, prevStepSize;
        //double prevTemp, prevFriction, prevErrorTol;
        RealOpenMM* projectionVectors;

};

} /* namespace OpenMM_LTMD */

#endif /* REFERENCE_INTEGRATE_NML_STEP_KERNEL_H */

