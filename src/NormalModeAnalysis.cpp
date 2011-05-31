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

#include "nmlopenmm/NormalModeAnalysis.h"
#include "openmm/OpenMMException.h"
#include "openmm/State.h"
#include "openmm/Vec3.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ForceImpl.h"
#include "jama_eig.h"
#include <algorithm>
#include <vector>
//#include <vecLib/clapack.h>

using namespace OpenMM;
using namespace std;

static void findEigenvaluesJama(const TNT::Array2D<float>& matrix, TNT::Array1D<float>& values, TNT::Array2D<float>& vectors) {
    JAMA::Eigenvalue<float> decomp(matrix);
    decomp.getRealEigenvalues(values);
    decomp.getV(vectors);
}
/*
static void findEigenvaluesLapack(const TNT::Array2D<float>& matrix, TNT::Array1D<float>& values, TNT::Array2D<float>& vectors) {
    long int n = matrix.dim1();
    char jobz = 'V';
    char uplo = 'U';
    long int lwork = 3*n-1;
    vector<float> a(n*n);
    vector<float> w(n);
    vector<float> work(lwork);
    long int info;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i*n+j] = matrix[i][j];
    ssyev_(&jobz, &uplo, &n, &a[0], &n, &w[0], &work[0], &lwork, &info);
    values = TNT::Array1D<float>(n);
    for (int i = 0; i < n; i++)
        values[i] = w[i];
    vectors = TNT::Array2D<float>(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            vectors[i][j] = a[i+j*n];
}
*/
void NormalModeAnalysis::computeEigenvectorsFull(ContextImpl& contextImpl, int numVectors) {
    Context& context = contextImpl.getOwner();
    bool isDoublePrecision = context.getPlatform().supportsDoublePrecision();
    State state = context.getState(State::Positions | State::Forces);
    vector<Vec3> positions = state.getPositions();
    System& system = context.getSystem();
    int numParticles = positions.size();
    int n = 3*numParticles;

    /*********************************************************************/
    /*                                                                   */
    /* Block Hessian Code (Cickovski/Sweet)                              */
    /*                                                                   */
    /*********************************************************************/

    TNT::Array2D<float> hessian(n,n); // Hessian matrix
                                      // Initial residue data (where in OpenMM?)
    int num_residues = 0;
    vector<int> blocks;
    for (int i = 0; i < numParticles; i++) {
       if (int(system.getParticleMass(i)) == 14) // N-terminus end (hack for now?)
          {
             num_residues++;
             blocks.push_back(i);
          }
    }
    
    cout << "Running block Hessian with " << num_residues << endl;
    // For now I am assuming residues per block is 1.
    // We can make this customizable later.
    
    
    
    
    

    /*********************************************************************/

    // Construct the mass weighted Hessian.
    


    TNT::Array2D<float> h(n, n);
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        for (int j = 0; j < 3; j++) {
            double delta = getDelta(positions[i][j], isDoublePrecision);
            positions[i][j] = pos[j]-delta;
            context.setPositions(positions);
            vector<Vec3> forces1 = context.getState(State::Forces).getForces();
            positions[i][j] = pos[j]+delta;
            context.setPositions(positions);
            vector<Vec3> forces2 = context.getState(State::Forces).getForces();
            positions[i][j] = pos[j];
            int col = i*3+j;
            int row = 0;
            for (int k = 0; k < numParticles; k++) {
                double scale = 1.0/(2*delta*sqrt(system.getParticleMass(i)*system.getParticleMass(k)));
                h[row++][col] = (forces1[k][0]-forces2[k][0])*scale;
                h[row++][col] = (forces1[k][1]-forces2[k][1])*scale;
                h[row++][col] = (forces1[k][2]-forces2[k][2])*scale;
            }
        }
    }

    // Make sure it is exactly symmetric.

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            float avg = 0.5f*(h[i][j]+h[j][i]);
            h[i][j] = avg;
            h[j][i] = avg;
        }
    }

    // Sort the eigenvectors by the absolute value of the eigenvalue.

    TNT::Array1D<float> d;
    TNT::Array2D<float> eigen;
    findEigenvaluesJama(h, d, eigen);
    //findEigenvaluesLapack(h, d, eigen);
    vector<pair<float, int> > sortedEigenvalues(n);
    for (int i = 0; i < n; i++)
        sortedEigenvalues[i] = make_pair(fabs(d[i]), i);
    sort(sortedEigenvalues.begin(), sortedEigenvalues.end());
    maxEigenvalue = sortedEigenvalues[n-1].first;

    // Record the eigenvectors.

    eigenvectors.resize(numVectors, vector<Vec3>(numParticles));
    for (int i = 0; i < numVectors; i++) {
        int index = sortedEigenvalues[i].second;
        for (int j = 0; j < numParticles; j++)
            eigenvectors[i][j] = Vec3(eigen[3*j][index], eigen[3*j+1][index], eigen[3*j+2][index]);
    }
}

void NormalModeAnalysis::computeEigenvectorsRestricting(ContextImpl& contextImpl, int numVectors) {
    Context& context = contextImpl.getOwner();
    bool isDoublePrecision = context.getPlatform().supportsDoublePrecision();
    State state = context.getState(State::Positions | State::Forces);
    vector<Vec3> positions = state.getPositions();
    vector<Vec3> forces = state.getForces();
    System& system = context.getSystem();
    int numParticles = positions.size();
    int n = 3*numParticles;

    // Construct the mass weighted Hessian.

    TNT::Array2D<float> h(n, n);
    for (int i = 0; i < numParticles; i++) {
        Vec3 pos = positions[i];
        for (int j = 0; j < 3; j++) {
            double delta = getDelta(positions[i][j], isDoublePrecision);
            positions[i][j] = pos[j]+delta;
            context.setPositions(positions);
            vector<Vec3> forces2 = context.getState(State::Forces).getForces();
            positions[i][j] = pos[j];
            int col = i*3+j;
            int row = 0;
            for (int k = 0; k < numParticles; k++) {
                double scale = 1.0/(delta*sqrt(system.getParticleMass(i)*system.getParticleMass(k)));
                h[row++][col] = (forces[k][0]-forces2[k][0])*scale;
                h[row++][col] = (forces[k][1]-forces2[k][1])*scale;
                h[row++][col] = (forces[k][2]-forces2[k][2])*scale;
            }
        }
    }

    // Make sure it is exactly symmetric.

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            float avg = 0.5f*(h[i][j]+h[j][i]);
            h[i][j] = avg;
            h[j][i] = avg;
        }
    }

    // Find a projection matrix to the smaller subspace.

    const vector<vector<int> >& molecules = contextImpl.getMolecules();
    buildTree(contextImpl);
    int m = 6*molecules.size()+bonds.size();
    projection.resize(m, vector<double>(n, 0.0));
    for (int i = 0; i < (int) molecules.size(); i++) {
        // Find the center of the molecule.

        const vector<int>& mol = molecules[i];
        Vec3 center;
        for (int j = 0; j < (int) mol.size(); j++)
            center += positions[mol[j]];
        center *= 1.0/mol.size();

        // Now loop over particles.

        for (int j = 0; j < (int) mol.size(); j++) {
            // Fill in the projection matrix.

            int particle = mol[j];
            projection[6*i][3*particle] = 1.0;
            projection[6*i+1][3*particle+1] = 1.0;
            projection[6*i+2][3*particle+2] = 1.0;
            Vec3 pos = positions[particle];
            projection[6*i+3][3*particle+1] = -(pos[2]-center[2]);
            projection[6*i+3][3*particle+2] = (pos[1]-center[1]);
            projection[6*i+4][3*particle+0] = (pos[2]-center[2]);
            projection[6*i+4][3*particle+2] = -(pos[0]-center[0]);
            projection[6*i+5][3*particle+0] = -(pos[1]-center[1]);
            projection[6*i+5][3*particle+1] = (pos[0]-center[0]);
        }
    }
    for (int i = 0; i < (int) bonds.size(); i++) {
        Vec3 base = positions[bonds[i].first];
        Vec3 dir = positions[bonds[i].second]-base;
        dir *= 1.0/sqrt(dir.dot(dir));
        vector<int> children;
        findChildren(*particleNodes[bonds[i].second], children);
        int row = 6*molecules.size()+i;
        for (int j = 0; j < (int) children.size(); j++) {
            int particle = children[j];
            Vec3 delta = dir.cross(positions[particle]-base);
            projection[row][3*particle] = delta[0];
            projection[row][3*particle+1] = delta[1];
            projection[row][3*particle+2] = delta[2];
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < numParticles; j++) {
            double scale = sqrt(system.getParticleMass(j));
            projection[i][3*j] *= scale;
            projection[i][3*j+1] *= scale;
            projection[i][3*j+2] *= scale;
        }
    }
    for (int i = 0; i < m; i++) {
        // Make this vector orthogonal to all previous ones.

        for (int j = 0; j < i; j++) {
            double dot = 0.0;
            for (int k = 0; k < n; k++)
                dot += projection[i][k]*projection[j][k];
            for (int k = 0; k < n; k++)
                projection[i][k] -= dot*projection[j][k];
        }

        // Normalize it.

        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += projection[i][j]*projection[i][j];
        double scale = 1.0/sqrt(sum);
        for (int j = 0; j < n; j++)
            projection[i][j] *= scale;
    }

    // Multiply by the projection matrix to get an m by m Hessian.

    vector<vector<double> > h2(m, vector<double>(n));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += projection[i][k]*h[k][j];
            h2[i][j] = sum;
        }
    TNT::Array2D<float> s(m, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += projection[i][k]*h2[j][k];
            s[i][j] = sum;
        }


    // Sort the eigenvectors by the absolute value of the eigenvalue.

    JAMA::Eigenvalue<float> decomp(s);
    TNT::Array1D<float> d;
    decomp.getRealEigenvalues(d);
    vector<pair<float, int> > sortedEigenvalues(m);
    for (int i = 0; i < m; i++)
        sortedEigenvalues[i] = make_pair(fabs(d[i]), i);
    sort(sortedEigenvalues.begin(), sortedEigenvalues.end());
    maxEigenvalue = sortedEigenvalues[m-1].first;

    // Record the eigenvectors.

    TNT::Array2D<float> eigen;
    decomp.getV(eigen);
    eigenvectors.resize(numVectors, vector<Vec3>(numParticles));
    for (int i = 0; i < numVectors; i++) {
        int index = sortedEigenvalues[i].second;
        for (int j = 0; j < n; j += 3) {
            Vec3 sum;
            for (int k = 0; k < m; k++) {
                sum[0] += projection[k][j]*eigen[k][index];
                sum[1] += projection[k][j+1]*eigen[k][index];
                sum[2] += projection[k][j+2]*eigen[k][index];
            }
            eigenvectors[i][j/3] = sum;
        }
    }
}

void NormalModeAnalysis::computeEigenvectorsDihedral(ContextImpl& contextImpl, int numVectors) {
    Context& context = contextImpl.getOwner();
    State state = context.getState(State::Positions | State::Forces);
    vector<Vec3> positions = state.getPositions();
    vector<Vec3> forces = state.getForces();
    int numParticles = positions.size();
    System& system = context.getSystem();
    const vector<vector<int> >& molecules = contextImpl.getMolecules();
    buildTree(contextImpl);
    int n = 3*numParticles;
    int m = 6*molecules.size()+bonds.size();
    projection.resize(m, vector<double>(n, 0.0));

    // First compute derivatives with respect to global translations and rotations.

    for (int i = 0; i < (int) molecules.size(); i++) {
        // Find the center of the molecule.

        const vector<int>& mol = molecules[i];
        Vec3 center;
        for (int j = 0; j < (int) mol.size(); j++)
            center += positions[mol[j]];
        center *= 1.0/mol.size();

        // Now loop over particles.

        for (int j = 0; j < (int) mol.size(); j++) {
            // Fill in the projection matrix.

            int particle = mol[j];
            projection[6*i][3*particle] = 1.0;
            projection[6*i+1][3*particle+1] = 1.0;
            projection[6*i+2][3*particle+2] = 1.0;
            Vec3 pos = positions[particle];
            projection[6*i+3][3*particle+1] = -(pos[2]-center[2]);
            projection[6*i+3][3*particle+2] = (pos[1]-center[1]);
            projection[6*i+4][3*particle+0] = (pos[2]-center[2]);
            projection[6*i+4][3*particle+2] = -(pos[0]-center[0]);
            projection[6*i+5][3*particle+0] = -(pos[1]-center[1]);
            projection[6*i+5][3*particle+1] = (pos[0]-center[0]);
        }
    }

    // Compute derivatives with respect to dihedrals.

    for (int i = 0; i < (int) bonds.size(); i++) {
        Vec3 base = positions[bonds[i].first];
        Vec3 dir = positions[bonds[i].second]-base;
        dir *= 1.0/sqrt(dir.dot(dir));
        vector<int> children;
        findChildren(*particleNodes[bonds[i].second], children);
        int row = 6*molecules.size()+i;
        for (int j = 0; j < (int) children.size(); j++) {
            int particle = children[j];
            Vec3 delta = dir.cross(positions[particle]-base);
            projection[row][3*particle] = delta[0];
            projection[row][3*particle+1] = delta[1];
            projection[row][3*particle+2] = delta[2];
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < numParticles; j++) {
            double scale = sqrt(system.getParticleMass(j));
            projection[i][3*j] *= scale;
            projection[i][3*j+1] *= scale;
            projection[i][3*j+2] *= scale;
        }
    }
    for (int i = 0; i < m; i++) {
        // Make this vector orthogonal to all previous ones.

        for (int j = 0; j < i; j++) {
            double dot = 0.0;
            for (int k = 0; k < n; k++)
                dot += projection[i][k]*projection[j][k];
            for (int k = 0; k < n; k++)
                projection[i][k] -= dot*projection[j][k];
        }

        // Normalize it.

        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += projection[i][j]*projection[i][j];
        double scale = 1.0/sqrt(sum);
        for (int j = 0; j < n; j++)
            projection[i][j] *= scale;
    }

    // Construct an m by n "Hessian like" matrix.

    vector<vector<double> > h(m, vector<double>(n));
    vector<Vec3> positions2(numParticles);
    for (int i = 0; i < m; i++) {
        double delta = sqrt(1e-7);
        for (int j = 0; j < numParticles; j++)
            for (int k = 0; k < 3; k++)
                positions2[j][k] = positions[j][k]+delta*projection[i][3*j+k];///sqrt(system.getParticleMass(j));
        context.setPositions(positions2);
        vector<Vec3> forces2 = context.getState(State::Forces).getForces();
        for (int j = 0; j < numParticles; j++) {
            double scale = 1.0/delta;
            h[i][3*j] = (forces[j][0]-forces2[j][0])*scale;
            h[i][3*j+1] = (forces[j][1]-forces2[j][1])*scale;
            h[i][3*j+2] = (forces[j][2]-forces2[j][2])*scale;
        }
    }

    // Multiply by the projection matrix to get an m by m Hessian.

    TNT::Array2D<float> s(m, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += projection[i][k]*h[j][k];
            s[i][j] = sum;
        }

    // Make sure it is exactly symmetric.

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < i; j++) {
            float avg = 0.5f*(s[i][j]+s[j][i]);
            s[i][j] = avg;
            s[j][i] = avg;
        }
    }

    // Sort the eigenvectors by the absolute value of the eigenvalue.

    JAMA::Eigenvalue<float> decomp(s);
    TNT::Array1D<float> d;
    decomp.getRealEigenvalues(d);
    vector<pair<float, int> > sortedEigenvalues(m);
    for (int i = 0; i < m; i++)
        sortedEigenvalues[i] = make_pair(fabs(d[i]), i);
    sort(sortedEigenvalues.begin(), sortedEigenvalues.end());
    maxEigenvalue = sortedEigenvalues[m-1].first;

    // Record the eigenvectors.

    TNT::Array2D<float> eigen;
    decomp.getV(eigen);
    eigenvectors.resize(numVectors, vector<Vec3>(numParticles));
    for (int i = 0; i < numVectors; i++) {
        int index = sortedEigenvalues[i].second;
        for (int j = 0; j < n; j += 3) {
            Vec3 sum;
            for (int k = 0; k < m; k++) {
                sum[0] += projection[k][j]*eigen[k][index];
                sum[1] += projection[k][j+1]*eigen[k][index];
                sum[2] += projection[k][j+2]*eigen[k][index];
            }
            eigenvectors[i][j/3] = sum;
        }
    }
}

double NormalModeAnalysis::getDelta(double value, bool isDoublePrecision) {
    double delta = sqrt(isDoublePrecision ? 1e-16 : 1e-7)*max(fabs(value), 0.1);
    volatile double temp = value+delta;
    delta = temp-value;
    return delta;
}

void NormalModeAnalysis::buildTree(ContextImpl& context) {
    System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    int numConstraints = system.getNumConstraints();
    vector<pair<int, int> > allBonds(numConstraints);
    for (int i = 0; i < numConstraints; i++) {
        double dist;
        system.getConstraintParameters(i, allBonds[i].first, allBonds[i].second, dist);
    }
    for (int i = 0; i < (int) context.getForceImpls().size(); i++) {
        const ForceImpl& force = *context.getForceImpls()[i];
        const vector<pair<int, int> >& forceBonds = force.getBondedParticles();
        for (int j = 0; j < (int) forceBonds.size(); j++)
            allBonds.push_back(forceBonds[j]);
    }
    particleBonds.resize(numParticles);
    for (int i = 0; i < (int) allBonds.size(); i++) {
        particleBonds[allBonds[i].first].push_back(allBonds[i].second);
        particleBonds[allBonds[i].second].push_back(allBonds[i].first);
    }



    vector<bool> processed(numParticles, false);
    for (int i = 0; i < numParticles; i++)
        if (!processed[i]) {
            treeRoots.push_back(TreeNode(i));
            processed[i] = true;
            processTreeNode(treeRoots.back(), processed, true);
        }
    for (int i = 0; i < (int) treeRoots.size(); i++)
        recordParticleNodes(treeRoots[i]);
}

void NormalModeAnalysis::processTreeNode(TreeNode& node, vector<bool>& processed, bool isRootNode) {
    vector<int>& bonded = particleBonds[node.particle];
    for (int i = 0; i < (int) bonded.size(); i++)
        if (!processed[bonded[i]]) {
            node.children.push_back(TreeNode(bonded[i]));
            processed[bonded[i]] = true;
            processTreeNode(node.children.back(), processed, false);
            node.totalChildren += node.children.back().totalChildren+1;
            if (node.children.back().totalChildren > 0 && !isRootNode)
                bonds.push_back(make_pair(node.particle, bonded[i]));
        }
}

void NormalModeAnalysis::recordParticleNodes(TreeNode& node) {
    particleNodes[node.particle] = &node;
    for (int i = 0; i < (int) node.children.size(); i++)
        recordParticleNodes(node.children[i]);
}

void NormalModeAnalysis::findChildren(const TreeNode& node, std::vector<int>& children) const {
    for (int i = 0; i < (int) node.children.size(); i++) {
        children.push_back(node.children[i].particle);
        findChildren(node.children[i], children);
    }
}

