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
#include "tnt_array2d_utils.h"
#include <algorithm>
#include <vector>
//#include <vecLib/clapack.h>
#include <fstream>
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

unsigned int NormalModeAnalysis::blockNumber(int p) {
    unsigned int block = 0;
    while (block != blocks.size() && blocks[block] <= p) block++;
    return block-1;
}

bool NormalModeAnalysis::inSameBlock(int p1, int p2, int p3=-1, int p4=-1) {
    if (blockNumber(p1) != blockNumber(p2)) return false;
    
    if (p3 != -1 && blockNumber(p3) != blockNumber(p1)) return false;
  
    if (p4 != -1 && blockNumber(p4) != blockNumber(p1)) return false;
 
    return true;   // They're all the same!
}

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


    /*********************************************************************/
   
    TNT::Array2D<float> hessian(n,n); // Hessian matrix (note n = 3N!)
                                      // Initial residue data (where in OpenMM?)
    
    // For now, since OpenMM input files do not contain residue information
    // I am assuming that they will always start with the N-terminus, just for testing.
    // This is true for the villin.xml but may not be true in the future.
    int num_residues = 0;
    for (int i = 0; i < numParticles; i++) {
       if (int(system.getParticleMass(i)) == 14) // N-terminus end, Nitrogen atom
          {
             num_residues++;
             blocks.push_back(i);
          }
    }
    cout << "Running block Hessian with " << num_residues << endl;
    cout << "Total number of blocks: " << blocks.size() << endl;
    // Creating a whole new system called the blockSystem.
    // This system will only contain bonds, angles, dihedrals, and impropers
    // between atoms in the same block. 
    // Also contains pairwise force terms which are zeroed out for atoms
    // in different blocks.
    // This necessitates some copying from the original system, but is required
    // because OpenMM populates all data when it reads XML.
    System* blockSystem = new System();

    // Copy all atoms into the block system.
    for (int i = 0; i < numParticles; i++) {
       blockSystem->addParticle(system.getParticleMass(i));
    } 
    
    // Copy the center of mass force.
    //blockSystem->addForce(&system.getForce(0));    

    // Create a new harmonic bond force.
    // This only contains pairs of atoms which are in the same block.
    // I have to iterate through each bond from the old force, then
    // selectively add them to the new force based on this condition.
    HarmonicBondForce hf;
    cout << "Number of forces: " << system.getNumForces() << endl;
    const HarmonicBondForce* ohf = dynamic_cast<const HarmonicBondForce*>(&system.getForce(1));
    for (int i = 0; i < ohf->getNumBonds(); i++) {
        // For our system, add bonds between atoms in the same block
        int particle1, particle2;
        double length, k;
        ohf->getBondParameters(i, particle1, particle2, length, k);
        if (inSameBlock(particle1, particle2)) {
	   cout << particle1 << " and " << particle2 << " are in the same block." << endl;
           hf.addBond(particle1, particle2, length, k);
        }
	else
	   cout << particle1 << " and " << particle2 << " are not in the same block." << endl;
    }
    blockSystem->addForce(&hf);


    // Same thing with the angle force....
    HarmonicAngleForce af;
    const HarmonicAngleForce* ahf = dynamic_cast<const HarmonicAngleForce*>(&system.getForce(2));
    for (int i = 0; i < ahf->getNumAngles(); i++) {
        // For our system, add bonds between atoms in the same block
        int particle1, particle2, particle3;
        double angle, k;
        ahf->getAngleParameters(i, particle1, particle2, particle3, angle, k);
        if (inSameBlock(particle1, particle2, particle3)) {
           af.addAngle(particle1, particle2, particle3, angle, k);
        }
    }
    blockSystem->addForce(&af);


    // And the dihedrals....
    PeriodicTorsionForce ptf;
    const PeriodicTorsionForce* optf = dynamic_cast<const PeriodicTorsionForce*>(&system.getForce(3));
    for (int i = 0; i < optf->getNumTorsions(); i++) {
        // For our system, add bonds between atoms in the same block
        int particle1, particle2, particle3, particle4, periodicity;
        double phase, k;
        optf->getTorsionParameters(i, particle1, particle2, particle3, particle4, periodicity, phase, k);
        if (inSameBlock(particle1, particle2, particle3, particle4)) {
           ptf.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k);
        }
    }
    blockSystem->addForce(&ptf);

    // And the impropers....
    RBTorsionForce rbtf;
    const RBTorsionForce* orbtf = dynamic_cast<const RBTorsionForce*>(&system.getForce(4));
    for (int i = 0; i < orbtf->getNumTorsions(); i++) {
        // For our system, add bonds between atoms in the same block
        int particle1, particle2, particle3, particle4;
        double c0, c1, c2, c3, c4, c5;
        orbtf->getTorsionParameters(i, particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5);
        if (inSameBlock(particle1, particle2, particle3, particle4)) {
           rbtf.addTorsion(particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5);
        }
    }
    blockSystem->addForce(&rbtf);


    // This is a custom nonbonded pairwise force and
    // includes terms for both LJ and Coulomb. 
    // Note that the step term will go to zero if block1 does not equal block 2,
    // and will be one otherwise.
    CustomNonbondedForce* customNonbonded = new CustomNonbondedForce("(step(block1-block2)*step(block2-block1))*(4*eps*((sigma/r)^12-(sigma/r)^6)+138.935456*q/r); q=q1*q2; sigma=0.5*(sigma1+sigma2); eps=sqrt(eps1*eps2)");
    const NonbondedForce* nbf = dynamic_cast<const NonbondedForce*>(&system.getForce(5));
    
 
    // To make a custom nonbonded force work, you have to add parameters.
    // The block number is new for this particular application, the other
    // three are copied from the old system.
    customNonbonded->addPerParticleParameter("block");
    customNonbonded->addPerParticleParameter("q");
    customNonbonded->addPerParticleParameter("sigma");
    customNonbonded->addPerParticleParameter("eps");
    
    vector<double> params(4);
    for (int i = 0; i < nbf->getNumParticles(); i++) {
        double charge, sigma, epsilon;
        nbf->getParticleParameters(i, charge, sigma, epsilon);
        params[0] = blockNumber(i);   // block #
        params[1] = charge;
        params[2] = sigma;
        params[3] = epsilon;
        customNonbonded->addParticle(params);
    }
 
    // Copy the exclusions.
    for (int i = 0; i < nbf->getNumExceptions(); i++) {
        int p1, p2;
        double cp, sig, eps;
        nbf->getExceptionParameters(i, p1, p2, cp, sig, eps);
        customNonbonded->addExclusion(p1, p2);
    }   

    // Copy the algorithm then add the force.
    customNonbonded->setNonbondedMethod((CustomNonbondedForce::NonbondedMethod)nbf->getNonbondedMethod());
    customNonbonded->setCutoffDistance((CustomNonbondedForce::NonbondedMethod)nbf->getCutoffDistance());
    blockSystem->addForce(customNonbonded);   


    // Copy the positions.
    Context blockContext(*blockSystem, context.getIntegrator());
    blockContext.setPositions(state.getPositions());
    
    /*********************************************************************/

    // Construct the mass weighted Hessian, and the block Hessian.
    // The latter should turn out to be a block Hessian
    // since appropriate forces have been zeroed out in a separate context
    // blockContext.
    // Finite difference method works the same, you perturb positions twice
    // and calculate forces each time, and you must scale by 1/2*dx*M^2.
    TNT::Array2D<float> h(n, n);
    cout << "Numparticles: " << numParticles << endl;
    for (int i = 0; i < numParticles; i++) {
        cout << i << endl;
        Vec3 pos = positions[i];
        for (int j = 0; j < 3; j++) {
	    // Block Hessian AND Hessian for now
	    double delta = getDelta(positions[i][j], isDoublePrecision);
            positions[i][j] = pos[j]-delta;
	    context.setPositions(positions);
            blockContext.setPositions(positions);
            vector<Vec3> forces1 = blockContext.getState(State::Forces).getForces();
            vector<Vec3> forces1full = context.getState(State::Forces).getForces();
            positions[i][j] = pos[j]+delta;
            blockContext.setPositions(positions);
            context.setPositions(positions);
            vector<Vec3> forces2 = blockContext.getState(State::Forces).getForces();
            vector<Vec3> forces2full = context.getState(State::Forces).getForces();
            positions[i][j] = pos[j];
            int col = i*3+j;
            int row = 0;
            for (int k = 0; k < numParticles; k++) {
                double scale = 1.0/(2*delta*sqrt(blockSystem->getParticleMass(i)*blockSystem->getParticleMass(k)));
                h[row++][col] = (forces1[k][0]-forces2[k][0])*scale;
                h[row++][col] = (forces1[k][1]-forces2[k][1])*scale;
                h[row++][col] = (forces1[k][2]-forces2[k][2])*scale;
		row = row - 3;
                hessian[row++][col] = (forces1full[k][0]-forces2full[k][0])*scale;
                hessian[row++][col] = (forces1full[k][1]-forces2full[k][1])*scale;
                hessian[row++][col] = (forces1full[k][2]-forces2full[k][2])*scale;
            }
        }
    }

    // Make sure it is exactly symmetric.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            float avg = 0.5f*(h[i][j]+h[j][i]);
            h[i][j] = avg;
            h[j][i] = avg;
            avg = 0.5f*(hessian[i][j]+hessian[j][i]);
            hessian[i][j] = avg;
            hessian[j][i] = avg;
        }
    }

    // Print the Hessian to a file.
    // Put both to a file.
    ofstream hess("hessian.txt", ios::out);
    cout << "PRINTING HESSIAN: " << endl << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            hess << "H(" << i << ", " << j << "): " << hessian[i][j] << endl;
        }
    }


    // Diagonalize each block Hessian, get Eigenvectors
    // Note: The eigenvalues will be placed in one large array, because
    //       we must sort them to get k
    vector<float> Di;
    vector<TNT::Array1D<float> > bigD;
    vector<TNT::Array2D<float> > bigQ;
    for (int i = 0; i < blocks.size(); i++) {
       cout << "Diagonalizing block: " << i << endl;   
 
       // 1. Determine the starting and ending index for the block
       //    This means that the upper left corner of the block will be at (startatom, startatom)
       //    And the lower right corner will be at (endatom, endatom)
       int startatom = 3*blocks[i];
       int endatom;
       if (i == blocks.size()-1)
          endatom = 3*numParticles-1;
       else
          endatom = 3*blocks[i+1] - 1; 

       // 2. Get the block Hessian Hii
       //    Right now I'm just doing a copy from the big Hessian
       //    There's probably a more efficient way but for now I just want things to work..
       TNT::Array2D<float> h_tilde(endatom-startatom+1, endatom-startatom+1);
       int xpos = 0;
       for (int j = startatom; j <= endatom; j++) {
          int ypos = 0;
          for (int k = startatom; k <= endatom; k++)
	     {
             h_tilde[xpos][ypos++] = hessian[j][k];
	     }
          xpos++;
       }
       
       // 3. Diagonalize the block Hessian only, and get eigenvectors
       TNT::Array1D<float> di(endatom-startatom+1);
       TNT::Array2D<float> Qi(endatom-startatom+1, endatom-startatom+1);
       findEigenvaluesJama(h_tilde, di, Qi);

       // 4. Copy eigenvalues to big array
       //    This is necessary because we have to sort them, and determine
       //    the cutoff eigenvalue for everybody.
       for (int j = 0; j < di.dim(); j++)
          Di.push_back(di[j]);

       // 5. Push eigenvectors into matrix
       bigD.push_back(di);
       bigQ.push_back(Qi);
    }

    cout << "Size of Di: " << Di.size() << endl;
    cout << "Size of bigD: " << bigD.size() << endl;
    cout << "Size of bigQ: " << bigQ.size() << endl;

    //***********************************************************
    // This section here is only to find the cuttoff eigenvalue.
    // First sort the eigenvectors by the absolute value of the eigenvalue.
    vector<pair<float, int> > sortedEvalues(Di.size());
    for (int i = 0; i < Di.size(); i++)
       sortedEvalues[i] = make_pair(fabs(Di[i]), i);
    sort(sortedEvalues.begin(), sortedEvalues.end()); 
    int bdof = 12;
    cout << "Number of eigenvalues is: " << sortedEvalues.size() << endl;
    float cutEigen = sortedEvalues[bdof*blocks.size()].first;  // This is the cutoff eigenvalue
    cout << "Cutoff eigenvalue is: " << cutEigen << endl;
    //***********************************************************

    // Build E.
    // For each Qi:
    //    Sort individual eigenvalues.
    //    Find some k such that k is the index of the largest eigenvalue less or equal to cutEigen
    //    Put those first k eigenvectors into E.
    vector<vector<float> > bigE;
    
    
    for (int i = 0; i < bigQ.size(); i++) {
        cout << "Putting eigenvector " << i << " into E" << endl;
        cout << "BigD dim: " << bigD[i].dim() << endl;
        vector<pair<float, int> > sE(bigD[i].dim());
        int k = 0;
        cout << "Finding k" << endl;   
        // Here we find k as the number of eigenvectors
        // smaller than the cutoff eigenvalue.
        // After we sort them, then k will be the index
        // of the smallest eigenvalue bigger than the cutoff value.
        for (int j = 0; j < bigD[i].dim(); j++) {
           sE[j] = make_pair(fabs(bigD[i][j]), j);
           if (bigD[i][j] <= cutEigen) k++;
        }
	cout << "Sorting Eigenvalues" << endl;
        sort(sE.begin(), sE.end());
 
        cout << "Putting k eigenvectors in.  k is: " << k << endl;
        // Put the eigenvectors in the corresponding order
        // into E.  Note that k is the index of the smallest
        // eigenvalue ABOVE the cutoff, so we have to put in the values
        // at indices 0 to k-1. 
	for (int a = 0; a < k; a++) {
           // Again, these are the corners of the block Hessian:
           // (startatom, startatom) and (endatom, endatom).
           int startatom = blocks[i]*3;
           int endatom;
           if (i == blocks.size()-1)
             endatom = 3*numParticles-1;
           else
             endatom = 3*blocks[i+1] - 1; 

           // This is an entry for matrix E. 
           // The eigenvectors will occupy row startatom through
           // row endatom.
           // Thus we must pad with zeroes before startatom and after
           // endatom.
           // Note that the way this is set up, bigE is actually E^T and not E,
           // since we form the vectors and THEN push.
           vector<float> entryE(n);
           int pos = 0;
           for (int j = 0; j < startatom; j++) // Pad beginning
              entryE[pos++] = 0;
           for (int j = 0; j < bigQ[i].dim2(); j++) // Eigenvector entries
              entryE[pos++] = bigQ[i][sE[a].second][j];  
           for (int j = endatom+1; j < n; j++)  // Pad end
              entryE[pos++] = 0;

           bigE.push_back(entryE);
        }
    }
    cout << "Size of bigE: " << bigE.size() << endl;
    

    // Inefficient, needs to improve.
    // Basically, just setting up E and E^T by
    // copying values from bigE.
    // Again, right now I'm only worried about
    // correctness plus this time will be marginal compared to
    // diagonalization.
    int m = bigE.size();
    TNT::Array2D<float> E(n, m);
    TNT::Array2D<float> E_transpose(m, n);
    for (int i = 0; i < m; i++)
       for (int j = 0; j < n; j++) {
          E[j][i] = bigE[i][j];
          E_transpose[i][j] = bigE[i][j];
       }

    // Compute S, which is equal to E^T * H * E.
    // Using the matmult function of Jama.
    cout << "Calculating S..." << endl;
    cout << "Dimensions of E transpose: " << E_transpose.dim1() << " x " << E_transpose.dim2() << endl;
    cout << "Dimensions of Hessian: " << hessian.dim1() << " x " << hessian.dim2() << endl;
    cout << "Dimensions of E: " << E.dim1() << " x " << E.dim2() << endl;
    TNT::Array2D<float> S(m, m);
    S = matmult(matmult(E_transpose, hessian), E);

    // Change to file
    cout << "PRINTING S: " << endl;
    for (unsigned int i = 0; i < S.dim1(); i++) {
       for (unsigned int j = 0; j < S.dim2(); j++) {
            float avg = 0.5f*(S[i][j]+S[j][i]);
            S[i][j] = avg;
            S[j][i] = avg;
          cout << S[i][j] << " ";
       }
       cout << endl;
    }
    
    // Diagonalizing S by finding eigenvalues and eigenvectors...
    cout << "Diagonalizing S..." << endl;
    TNT::Array1D<float> dS;
    TNT::Array2D<float> q;
    findEigenvaluesJama(S, dS, q);
    

    // Sort by ABSOLUTE VALUE of eigenvalues.
    sortedEvalues.clear();
    sortedEvalues.resize(dS.dim());
    for (int i = 0; i < dS.dim(); i++)
       sortedEvalues[i] = make_pair(fabs(dS[i]), i);
    sort(sortedEvalues.begin(), sortedEvalues.end()); 
    
    TNT::Array2D<float> Q_transpose(q.dim1(), q.dim2());
    TNT::Array2D<float> Q(q.dim2(), q.dim1());
    for (int i = 0; i < sortedEvalues.size(); i++)
       for (int j = 0; j < q.dim2(); j++)
          Q_transpose[i][j] = q[sortedEvalues[i].second][j];
    maxEigenvalue = sortedEvalues[dS.dim()-1].first;
    
    for (int i = 0; i < q.dim2(); i++)
       for (int j = 0; j < q.dim1(); j++)
           Q[i][j] = Q_transpose[j][i];


    // Compute U, set of approximate eigenvectors.
    // BUG: Q should be sorted before multiplying by E.
    // U = E*Q.
    cout << "Computing U..." << endl;
    TNT::Array2D<float> U = matmult(E, Q); //E*Q;

    // Record the eigenvectors.
    // These will be placed in a file eigenvectors.txt
    cout << "Computing final eigenvectors... " << endl;
    ofstream outfile("eigenvectors.txt", ios::out);
    eigenvectors.resize(numVectors, vector<Vec3>(numParticles));
    for (int i = 0; i < numVectors; i++) {
        for (int j = 0; j < numParticles; j++) {
            eigenvectors[i][j] = Vec3(U[3*j][i], U[3*j+1][i], U[3*j+2][i]);
            outfile << U[3*j][i] << " " << U[3*j+1][i] << " " << U[3*j+2][i] << endl;
        }
    }

    // Record the eigenvalues.
    // These will be placed in a file eigenvalues.txt
    ofstream evalfile("eigenvalues.txt", ios::out);
    for (int i = 0; i < sortedEvalues.size(); i++)
       evalfile << i << " " << sortedEvalues[i].first << endl;

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

