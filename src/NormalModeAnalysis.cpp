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
#include "nmlopenmm/NMLIntegrator.h"
#include "openmm/OpenMMException.h"
#include "openmm/State.h"
#include "openmm/Vec3.h"
#include <sys/time.h>
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ForceImpl.h"
#include "jama_eig.h"
#include "tnt_array2d_utils.h"
#include <algorithm>
#include <vector>
#include <fstream>
//#include "/afs/crc.nd.edu/user/t/tcickovs/clapack-3.2.1-CMAKE/INCLUDE/clapack.h"
using namespace OpenMM;
using namespace OpenMM_LTMD;
using namespace std;

extern "C" void ssyev_( char *jobz, char *uplo, int *n, double *a, int *lda,
//void ssyev_( char *jobz, char *uplo, int *n, double *a, int *lda,
        double *w, double *work, int *lwork, int *info );

extern "C" int f2c_dgemm (char* transa, char* transb, int* m, int* n, int* k, double* alpha,
                        double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);

static void findEigenvaluesJama(const TNT::Array2D<float>& matrix, TNT::Array1D<float>& values, TNT::Array2D<float>& vectors) {
    JAMA::Eigenvalue<float> decomp(matrix);
    decomp.getRealEigenvalues(values);
    decomp.getV(vectors);
}

static void findEigenvaluesLapack(const TNT::Array2D<float>& matrix, TNT::Array1D<float>& values, TNT::Array2D<float>& vectors) {
    int n = matrix.dim1();
    //long int n = matrix.dim1();
    char jobz = 'V';
    char uplo = 'U';
    int lwork = 3*n-1;
    vector<double> a(n*n);
    vector<double> w(n);
    vector<double> work(lwork);
    int info;
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

static void matMultLapack(const TNT::Array2D<float>& matrix1, TNT::Array2D<float>& matrix2, TNT::Array2D<float>& matrix3) {
   int m = matrix1.dim1();
   int k = matrix1.dim2();
   int n = matrix2.dim2();
   char transa = 'N';
   char transb = 'N';
   double alpha = 0.0;
   double beta = 0.0;
   int lda = m;
   int ldb = k;
   int ldc = m;
   vector<double> a(m*k);
   vector<double> b(k*n);
   vector<double> c(m*n);
   cout << "LDA: " << lda << " LDB: " << ldb << " LDC: " << ldc << endl;
   for (int i = 0; i < m; i++)
       for (int j = 0; j < k; j++)
           a[i*k+j] = matrix1[i][j];

   for (int i = 0; i < k; i++)
       for (int j = 0; j < n; j++)
           b[i*n+j] = matrix2[i][j];

   f2c_dgemm(&transa, &transb, &m, &n, &k, &alpha, &a[0], &lda, &b[0], &ldb, &beta, &c[0], &ldc);

   for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
          matrix3[i][j] = c[i*n+j];
}


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

void NormalModeAnalysis::computeEigenvectorsFull(ContextImpl& contextImpl, int numVectors,
                                                 LTMDParameters* ltmd) {
    struct timeval tp_begin;
    struct timeval tp_hess;
    struct timeval tp_diag;
    struct timeval tp_e;
    struct timeval tp_s;
    struct timeval tp_q;
    struct timeval tp_u;
    struct timeval tp_end;
    
    gettimeofday(&tp_begin, NULL);
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
    int res_per_block = 1;
    int first_atom = 0;
    int flag = 0;
    int largest_block_size = -1; // Keep track of the largest block size, we'll
                                 // need it to parallelize.
    System* blockSystem = new System();
    int pos = 0;
    int rescount = 0;
    blocks.push_back(0);  // Starting atom starts first block
    for (int i = 0; i < numParticles; i++) {
       blockSystem->addParticle(system.getParticleMass(i));
       if (i-blocks[pos] == ltmd->residue_sizes[pos] && rescount < ltmd->res_per_block)
       {
           cout << "Atom: " << i << endl;
           blocks.push_back(i);
	   if (ltmd->residue_sizes[pos] > largest_block_size)
	       largest_block_size = ltmd->residue_sizes[pos];
           pos++;
	   rescount++;
	   if (rescount == ltmd->res_per_block)
	      rescount = 0;
       }

       /*if (int(system.getParticleMass(i)) == 14) // N-terminus end, Nitrogen atom
          {
	     if (flag == 0) {
                num_residues++;
		cout << "Atom: " << i << endl;
                blocks.push_back(i);
		if (blocks.size() > 1) {
		   int blocksize = blocks[blocks.size()-1] - blocks[blocks.size()-2];
                   if (blocksize > largest_block_size)
		      largest_block_size = blocksize;
		}
		flag++;
	     }
	     else
	        flag++;
	     if (flag == res_per_block)
	       flag = 0;
          }*/
    }
    
    // Creating a whole new system called the blockSystem.
    // This system will only contain bonds, angles, dihedrals, and impropers
    // between atoms in the same block. 
    // Also contains pairwise force terms which are zeroed out for atoms
    // in different blocks.
    // This necessitates some copying from the original system, but is required
    // because OpenMM populates all data when it reads XML.
    // Copy all atoms into the block system.
    
    // Copy the center of mass force.
    cout << "adding forces..." << endl;
    for (int i = 0; i < ltmd->forces.size(); i++) {
       string forcename = ltmd->forces[i].name;
       cout << "Adding force " << forcename << " at index " << ltmd->forces[i].index << endl;
       if (forcename == "CenterOfMass") 
          blockSystem->addForce(&system.getForce(ltmd->forces[i].index));    
       else if (forcename == "Bond") {
	  // Create a new harmonic bond force.
          // This only contains pairs of atoms which are in the same block.
          // I have to iterate through each bond from the old force, then
          // selectively add them to the new force based on this condition.
          HarmonicBondForce* hf = new HarmonicBondForce();
          const HarmonicBondForce* ohf = dynamic_cast<const HarmonicBondForce*>(&system.getForce(ltmd->forces[i].index));
          for (int i = 0; i < ohf->getNumBonds(); i++) {
             // For our system, add bonds between atoms in the same block
             int particle1, particle2;
             double length, k;
             ohf->getBondParameters(i, particle1, particle2, length, k);
             if (inSameBlock(particle1, particle2)) {
                hf->addBond(particle1, particle2, length, k);
             }
          }
          blockSystem->addForce(hf);
       }
       else if (forcename == "Angle") {
          // Same thing with the angle force....
          HarmonicAngleForce* af = new HarmonicAngleForce();
          const HarmonicAngleForce* ahf = dynamic_cast<const HarmonicAngleForce*>(&system.getForce(ltmd->forces[i].index));
          for (int i = 0; i < ahf->getNumAngles(); i++) {
             // For our system, add bonds between atoms in the same block
             int particle1, particle2, particle3;
             double angle, k;
             ahf->getAngleParameters(i, particle1, particle2, particle3, angle, k);
             if (inSameBlock(particle1, particle2, particle3)) {
                af->addAngle(particle1, particle2, particle3, angle, k);
             }
          }
          blockSystem->addForce(af);
       }
       else if (forcename == "Dihedral") {
          // And the dihedrals....
          PeriodicTorsionForce* ptf = new PeriodicTorsionForce();
          const PeriodicTorsionForce* optf = dynamic_cast<const PeriodicTorsionForce*>(&system.getForce(ltmd->forces[i].index));
          for (int i = 0; i < optf->getNumTorsions(); i++) {
             // For our system, add bonds between atoms in the same block
             int particle1, particle2, particle3, particle4, periodicity;
             double phase, k;
             optf->getTorsionParameters(i, particle1, particle2, particle3, particle4, periodicity, phase, k);
             if (inSameBlock(particle1, particle2, particle3, particle4)) {
                ptf->addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k);
             }
          }
          blockSystem->addForce(ptf);
       }
       else if (forcename == "Improper") {
          // And the impropers....
          RBTorsionForce* rbtf = new RBTorsionForce();
          const RBTorsionForce* orbtf = dynamic_cast<const RBTorsionForce*>(&system.getForce(ltmd->forces[i].index));
          for (int i = 0; i < orbtf->getNumTorsions(); i++) {
             // For our system, add bonds between atoms in the same block
             int particle1, particle2, particle3, particle4;
             double c0, c1, c2, c3, c4, c5;
             orbtf->getTorsionParameters(i, particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5);
             if (inSameBlock(particle1, particle2, particle3, particle4)) {
                rbtf->addTorsion(particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5);
             }
          }
          blockSystem->addForce(rbtf);
       }
       else if (forcename == "Nonbonded") {
          // This is a custom nonbonded pairwise force and
          // includes terms for both LJ and Coulomb. 
          // Note that the step term will go to zero if block1 does not equal block 2,
          // and will be one otherwise.
          const NonbondedForce* nbf = dynamic_cast<const NonbondedForce*>(&system.getForce(ltmd->forces[i].index));
          NonbondedForce* nonbonded = new NonbondedForce();
 
          for (int i = 0; i < nbf->getNumParticles(); i++) {
             double charge, sigma, epsilon;
             nbf->getParticleParameters(i, charge, sigma, epsilon);
             nonbonded->addParticle(charge, sigma, epsilon);
          }
 
          // Copy the exclusions.
          for (int i = 0; i < nbf->getNumExceptions(); i++) {
             int p1, p2;
             double cp, sig, eps;
             nbf->getExceptionParameters(i, p1, p2, cp, sig, eps);
	     if (inSameBlock(p1, p2))
                nonbonded->addException(p1, p2, cp, sig, eps);
          }   

          // Exclude interactions between atoms not in the same blocks
          for(int i = 0; i < nbf->getNumParticles(); i++)
	        for(int j = i + 1; j < nbf->getNumParticles(); j++)
                   if(!inSameBlock(i, j))
	              nonbonded->addException(i, j, 0.0, 0.0, 0.0);
          nonbonded->setNonbondedMethod(nbf->getNonbondedMethod());
          nonbonded->setCutoffDistance(nbf->getCutoffDistance());
          blockSystem->addForce(nonbonded);
       }
       else {
          cout << "Unknown Force: " << forcename << endl;
       }
    }
    cout << "done." << endl;

    // Copy the positions.
    NMLIntegrator integ(300, 100.0, 0.05);
    integ.setMaxEigenvalue(5e5);
    if (blockContext) delete blockContext;
    blockContext = new Context(*blockSystem, integ, Platform::getPlatformByName("Cuda"));
    bool isBlockDoublePrecision = blockContext->getPlatform().supportsDoublePrecision();
    vector<Vec3> blockPositions;
    for (int i = 0; i < numParticles; i++) {
       Vec3 atom(state.getPositions()[i][0], state.getPositions()[i][1], state.getPositions()[i][2]);
       blockPositions.push_back(atom);
    }

    blockContext->setPositions(blockPositions);
    /*********************************************************************/


    TNT::Array2D<float> h(n, n);
    largest_block_size *= 3; // degrees of freedom in the largest block
    vector<Vec3> tmp(blockPositions);
    vector<Vec3> tmp2(blockPositions);
    for (int i = 0; i < largest_block_size; i++)
       {
          vector<double> deltas(blocks.size());
          // Perturb the ith degree of freedom in EACH block
	  // Note: not all blocks will have i degrees, we have to check for this
          for (int j = 0; j < blocks.size(); j++) {
             int dof_to_perturb = 3*blocks[j]+i;
             int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

	     // Cases to not perturb, in this case just skip the block
	     if (j == blocks.size()-1 && atom_to_perturb >= numParticles) continue;
             if (j != blocks.size()-1 && atom_to_perturb >= blocks[j+1]) continue;
             
	     double blockDelta = getDelta(blockPositions[atom_to_perturb][dof_to_perturb % 3], isBlockDoublePrecision, ltmd);
	     deltas[j] = blockDelta;
             blockPositions[atom_to_perturb][dof_to_perturb % 3] = tmp[atom_to_perturb][dof_to_perturb % 3] - blockDelta;

	  }

          blockContext->setPositions(blockPositions);
          vector<Vec3> forces1 = blockContext->getState(State::Forces).getForces();
 

          
	  // Now, do it again...
          for (int j = 0; j < blocks.size(); j++) {
             int dof_to_perturb = 3*blocks[j]+i;
             int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

	     // Cases to not perturb, in this case just skip the block
	     if (j == blocks.size()-1 && atom_to_perturb >= numParticles) continue;
             if (j != blocks.size()-1 && atom_to_perturb >= blocks[j+1]) continue;
             
	     double blockDelta = deltas[j];
             blockPositions[atom_to_perturb][dof_to_perturb % 3] = tmp2[atom_to_perturb][dof_to_perturb % 3] + blockDelta;


	  }

          blockContext->setPositions(blockPositions);
          vector<Vec3> forces2 = blockContext->getState(State::Forces).getForces();
          blockContext->setPositions(tmp2);


          for (int j = 0; j < blocks.size(); j++) {
             int dof_to_perturb = 3*blocks[j]+i;
             int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

	     // Cases to not perturb, in this case just skip the block
	     if (j == blocks.size()-1 && atom_to_perturb >= numParticles) continue;
             if (j != blocks.size()-1 && atom_to_perturb >= blocks[j+1]) continue;
             
	     int col = (atom_to_perturb*3)+(dof_to_perturb % 3);
	     int row = 0;

             double blockDelta = deltas[j];
	     for (int k = 0; k < numParticles; k++) {
                double blockscale = 1.0/(2*blockDelta*sqrt(system.getParticleMass(atom_to_perturb)*system.getParticleMass(k)));
                h[row++][col] = (forces1[k][0]-forces2[k][0])*blockscale;
                h[row++][col] = (forces1[k][1]-forces2[k][1])*blockscale;
                h[row++][col] = (forces1[k][2]-forces2[k][2])*blockscale;
              }
	  }

       }


    // Construct the mass weighted Hessian, and the block Hessian.
    // The latter should turn out to be a block Hessian
    // since appropriate forces have been zeroed out in a separate context
    // blockContext.
    // Finite difference method works the same, you perturb positions twice
    // and calculate forces each time, and you must scale by 1/2*dx*M^2.
    //TNT::Array2D<float> h(n, n);
    /*for (int i = 0; i < numParticles; i++) {
        //Vec3 pos = positions[i];
        Vec3 blockpos = blockPositions[i];
	for (int j = 0; j < 3; j++) {
	    // Block Hessian AND Hessian for now
	    //double delta = getDelta(positions[i][j], isDoublePrecision);
	    double blockDelta = getDelta(blockPositions[i][j], isBlockDoublePrecision);
	    //positions[i][j] = pos[j]-delta;
	    blockPositions[i][j] = blockpos[j]-blockDelta;
	    //context.setPositions(positions);
            blockContext->setPositions(blockPositions);
            vector<Vec3> forces1 = blockContext->getState(State::Forces).getForces();
            //vector<Vec3> forces1full = context.getState(State::Forces).getForces();
            //positions[i][j] = pos[j]+delta;
	    blockPositions[i][j] = blockpos[j]+blockDelta;
            blockContext->setPositions(blockPositions);
            //context.setPositions(positions);
            vector<Vec3> forces2 = blockContext->getState(State::Forces).getForces();
            //vector<Vec3> forces2full = context.getState(State::Forces).getForces();
            //positions[i][j] = pos[j];
	    blockPositions[i][j] = blockpos[j];
            int col = i*3+j;
            int row = 0;
	    for (int k = 0; k < numParticles; k++) {
                //double scale = 1.0/(2*delta*sqrt(system.getParticleMass(i)*system.getParticleMass(k)));
                double blockscale = 1.0/(2*blockDelta*sqrt(system.getParticleMass(i)*system.getParticleMass(k)));
                h[row++][col] = (forces1[k][0]-forces2[k][0])*blockscale;
                //hessian[row++][col] = (forces1full[k][0]-forces2full[k][0])*scale;
                h[row++][col] = (forces1[k][1]-forces2[k][1])*blockscale;
                //hessian[row++][col] = (forces1full[k][1]-forces2full[k][1])*scale;
                h[row++][col] = (forces1[k][2]-forces2[k][2])*blockscale;
                //hessian[row++][col] = (forces1full[k][2]-forces2full[k][2])*scale;
            }
        }
    }*/

    // Make sure it is exactly symmetric.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            float avg = 0.5f*(h[i][j]+h[j][i]);
            h[i][j] = avg;
            h[j][i] = avg;
            //avg = 0.5f*(hessian[i][j]+hessian[j][i]);
            //hessian[i][j] = avg;
            //hessian[j][i] = avg;
        }
    }

    gettimeofday(&tp_hess, NULL);
    cout << "Time to compute hessian: " << (tp_hess.tv_sec - tp_begin.tv_sec) << endl;


    cout << "Diagonalizing..." << endl;

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
             h_tilde[xpos][ypos++] = h[j][k];
	     }
          xpos++;
       }
       
       // 3. Diagonalize the block Hessian only, and get eigenvectors
       TNT::Array1D<float> di(endatom-startatom+1);
       TNT::Array2D<float> Qi(endatom-startatom+1, endatom-startatom+1);
       findEigenvaluesJama(h_tilde, di, Qi);
       //findEigenvaluesLapack(h_tilde, di, Qi);

       // 4. Copy eigenvalues to big array
       //    This is necessary because we have to sort them, and determine
       //    the cutoff eigenvalue for everybody.
       for (int j = 0; j < di.dim(); j++)
          Di.push_back(di[j]);

       // 5. Push eigenvectors into matrix
       bigD.push_back(di);
       bigQ.push_back(Qi);
    }

    gettimeofday(&tp_diag, NULL);
    cout << "Time to diagonalize block hessian: " << (tp_diag.tv_sec - tp_hess.tv_sec) << endl;


    //***********************************************************
    // This section here is only to find the cuttoff eigenvalue.
    // First sort the eigenvectors by the absolute value of the eigenvalue.
    vector<pair<float, int> > sortedEvalues(Di.size());
    for (int i = 0; i < Di.size(); i++)
       sortedEvalues[i] = make_pair(fabs(Di[i]), i);
    sort(sortedEvalues.begin(), sortedEvalues.end()); 
    //int bdof = 12;
    float cutEigen = sortedEvalues[ltmd->bdof*blocks.size()].first;  // This is the cutoff eigenvalue

    // For each Qi:
    //    Sort individual eigenvalues.
    //    Find some k such that k is the index of the largest eigenvalue less or equal to cutEigen
    //    Put those first k eigenvectors into E.
    vector<vector<float> > bigE;
    
    for (int i = 0; i < bigQ.size(); i++) {
	vector<pair<float, int> > sE(bigD[i].dim());
        int k = 0;
        // Here we find k as the number of eigenvectors
        // smaller than the cutoff eigenvalue.
        // After we sort them, then k will be the index
        // of the smallest eigenvalue bigger than the cutoff value.
        for (int j = 0; j < bigD[i].dim(); j++) {
           sE[j] = make_pair(fabs(bigD[i][j]), j);
           if (bigD[i][j] <= cutEigen) k++;
        }
        sort(sE.begin(), sE.end());
 

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
              entryE[pos++] = bigQ[i][j][sE[a].second];  
           for (int j = endatom+1; j < n; j++)  // Pad end
              entryE[pos++] = 0;

           bigE.push_back(entryE);
        }
    }
    

    // Inefficient, needs to improve.
    // Basically, just setting up E and E^T by
    // copying values from bigE.
    // Again, right now I'm only worried about
    // correctness plus this time will be marginal compared to
    // diagonalization.
    int m = bigE.size();
    cout << "M: " << m << endl;
    TNT::Array2D<float> E(n, m);
    //TNT::Array2D<float> E_transpose(m, n);
    TNT::Array2D<float> EPS(n, m);
    TNT::Array2D<float> EPS_transpose(m, n);
    for (int i = 0; i < m; i++)
       for (int j = 0; j < n; j++) {
          E[j][i] = bigE[i][j];
          //E[j][i] = E_transpose[i][j] = bigE[i][j];
	  EPS[j][i] = EPS_transpose[i][j] = (1.0/sqrt(system.getParticleMass(i/3)))*bigE[i][j];
       }
    gettimeofday(&tp_e, NULL);
    cout << "Time to compute E: " << (tp_e.tv_sec - tp_diag.tv_sec) << endl;

    //*****************************************************************
    // Compute S, which is equal to E^T * H * E.
    // Using the matmult function of Jama.
    TNT::Array2D<float> S(m, m);
    
    //for (unsigned int i = 0; i < n; i++)
    //   for (unsigned int j = 0; j < m; j++) 
    //       EPS[i][j] = EPS_transpose[j][i] = (1.0/sqrt(system.getParticleMass(i/3)))*E[i][j];
    

    // Compute F(x0). 
    vector<Vec3> fx0 = context.getState(State::Forces).getForces();
    // Compute eps.
    double eps;

    // Make a temp copy of positions.
    vector<Vec3> tmppos(positions);

    // Loop over i.
    for (unsigned int k = 0; k < m; k++) {
       // Perturb positions.
       int pos = 0;
       for (unsigned int i = 0; i < numParticles; i++) {
          for (unsigned int j = 0; j < 3; j++) {
             positions[i][j] += eps*EPS[pos][k];
	     eps = getDelta(positions[i][j], isDoublePrecision, ltmd);
	     pos++;
	  }
       }
       context.setPositions(positions);

       // Calculate F(xi).
       vector<Vec3> fxi = context.getState(State::Forces).getForces();

       TNT::Array2D<float> Force_diff(n, 1);
       for (int i = 0; i < n; i++) {
           Force_diff[i][0] = fxi[i/3][i%3] - fx0[i/3][i%3];
	   }

       TNT::Array2D<float> Si(m, 1);
       //matMultLapack(EPS_transpose,Force_diff,Si);
       Si = matmult(EPS_transpose,Force_diff);

       // Copy to S.
       for (int i = 0; i < m; i++)
          S[i][k] = Si[i][0]*(1.0/eps);

       for (unsigned int i = 0; i < numParticles; i++) {
          for (unsigned int j = 0; j < 3; j++) {
             positions[i][j] = tmppos[i][j];
	  }
	  }
       context.setPositions(positions);
    }

    //*****************************************************************
    

    //S = matmult(matmult(E_transpose, hessian), E);

    // Change to file
    for (unsigned int i = 0; i < S.dim1(); i++) {
       for (unsigned int j = 0; j < S.dim2(); j++) {
            float avg = 0.5f*(S[i][j]+S[j][i]);
            S[i][j] = avg;
            S[j][i] = avg;
       }
    }
    gettimeofday(&tp_s, NULL);
    cout << "Time to compute S: " << (tp_s.tv_sec - tp_e.tv_sec) << endl;
    // Diagonalizing S by finding eigenvalues and eigenvectors...
    TNT::Array1D<float> dS;
    TNT::Array2D<float> q;
    findEigenvaluesJama(S, dS, q);
    //findEigenvaluesLapack(S, dS, q);
    

    // Sort by ABSOLUTE VALUE of eigenvalues.
    sortedEvalues.clear();
    sortedEvalues.resize(dS.dim());
    for (int i = 0; i < dS.dim(); i++)
       sortedEvalues[i] = make_pair(fabs(dS[i]), i);
    sort(sortedEvalues.begin(), sortedEvalues.end()); 
    
    //TNT::Array2D<float> Q_transpose(q.dim1(), q.dim2());
    TNT::Array2D<float> Q(q.dim2(), q.dim1());
    for (int i = 0; i < sortedEvalues.size(); i++)
       for (int j = 0; j < q.dim2(); j++) {
          Q[j][i] = q[j][sortedEvalues[i].second];	  
       }
    maxEigenvalue = sortedEvalues[dS.dim()-1].first;
    gettimeofday(&tp_q, NULL);
    cout << "Time to compute Q: " << (tp_q.tv_sec - tp_s.tv_sec) << endl;
    

    // Compute U, set of approximate eigenvectors.
    // U = E*Q.
    TNT::Array2D<float> U = matmult(E, Q); //E*Q;
    gettimeofday(&tp_u, NULL);
    cout << "Time to compute U: " << (tp_u.tv_sec - tp_q.tv_sec) << endl;

    // Record the eigenvectors.
    // These will be placed in a file eigenvectors.txt
    ofstream outfile("eigenvectors.txt", ios::out);
    eigenvectors.resize(numVectors, vector<Vec3>(numParticles));
    for (int i = 0; i < numVectors; i++) {
        for (int j = 0; j < numParticles; j++) {
            eigenvectors[i][j] = Vec3(U[3*j][i], U[3*j+1][i], U[3*j+2][i]);
            outfile << U[3*j][i] << " " << U[3*j+1][i] << " " << U[3*j+2][i] << endl;
        }
    }


    gettimeofday(&tp_end, NULL);
    cout << "Overall diagonalization time in seconds: " << (tp_end.tv_sec - tp_begin.tv_sec) << endl;
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
            double delta = getDelta(positions[i][j], isDoublePrecision, NULL);
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

double NormalModeAnalysis::getDelta(double value, bool isDoublePrecision, LTMDParameters* ltmd) {
    double delta = sqrt(ltmd->delta)*max(fabs(value), 0.1);
    //double delta = sqrt(isDoublePrecision ? 1e-16 : 1e-7)*max(fabs(value), 0.1);
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

