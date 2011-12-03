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
#include <sstream>

#include "OpenMM.h"
#include "LTMD/Analysis.h"
#include "LTMD/Integrator.h"


namespace OpenMM {
	namespace LTMD {

		template<typename T>
		static void findEigenvaluesJama( const TNT::Array2D<T>& matrix, TNT::Array1D<T>& values, TNT::Array2D<T>& vectors ) {
			JAMA::Eigenvalue<T> decomp( matrix );
			decomp.getRealEigenvalues( values );
			decomp.getV( vectors );
		}

		/*
		static void findEigenvaluesJama( const TNT::Array2D<double>& matrix, TNT::Array1D<double>& values, TNT::Array2D<double>& vectors ) {
			JAMA::Eigenvalue<double> decomp( matrix );
			decomp.getRealEigenvalues( values );
			decomp.getV( vectors );
		}
		*/

		/*
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
		*/

		unsigned int Analysis::blockNumber( int p ) {
			unsigned int block = 0;
			while( block != blocks.size() && blocks[block] <= p ) {
				block++;
			}
			return block - 1;
		}

		bool Analysis::inSameBlock( int p1, int p2, int p3 = -1, int p4 = -1 ) {
			if( blockNumber( p1 ) != blockNumber( p2 ) ) {
				return false;
			}

			if( p3 != -1 && blockNumber( p3 ) != blockNumber( p1 ) ) {
				return false;
			}

			if( p4 != -1 && blockNumber( p4 ) != blockNumber( p1 ) ) {
				return false;
			}

			return true;   // They're all the same!
		}

		void Analysis::computeEigenvectorsFull( ContextImpl &contextImpl, Parameters *ltmd ) {
			struct timeval tp_begin;
			struct timeval tp_hess;
			struct timeval tp_diag;
			struct timeval tp_e;
			struct timeval tp_s;
			struct timeval tp_q;
			struct timeval tp_u;
			struct timeval tp_end;

			gettimeofday( &tp_begin, NULL );
			Context &context = contextImpl.getOwner();
			bool isDoublePrecision = context.getPlatform().supportsDoublePrecision();
			State state = context.getState( State::Positions | State::Forces );
			vector<Vec3> positions = state.getPositions();
			System &system = context.getSystem();
			int numParticles = positions.size();
			int n = 3 * numParticles;
			int numVectors = ltmd->modes;

			/*********************************************************************/
			/*                                                                   */
			/* Block Hessian Code (Cickovski/Sweet)                              */
			/*                                                                   */
			/*********************************************************************/


			/*********************************************************************/

			//TNT::Array2D<float> hessian(n,n); // Hessian matrix (note n = 3N!)
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
			System *blockSystem = new System();
			int pos = 0;
			int rescount = 0;
			cout << "res per block " << ltmd->res_per_block << endl;
			for( int i = 0; i < numParticles; i++ ) {
				blockSystem->addParticle( system.getParticleMass( i ) );
			}

			int block_start = 0;
			for( int i = 0; i < ltmd->residue_sizes.size(); i++ ) {
				if( i % ltmd->res_per_block == 0 ) {
					blocks.push_back( block_start );
				}
				block_start += ltmd->residue_sizes[i];
			}

			for( int i = 1; i < blocks.size(); i++ ) {
				int block_size = blocks[i] - blocks[i - 1];
				if( block_size > largest_block_size ) {
					largest_block_size = block_size;
				}
			}

			cout << "blocks " << blocks.size() << endl;
			cout << blocks[blocks.size() - 1] << endl;

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
			for( int i = 0; i < ltmd->forces.size(); i++ ) {
				string forcename = ltmd->forces[i].name;
				cout << "Adding force " << forcename << " at index " << ltmd->forces[i].index << endl;
				if( forcename == "CenterOfMass" ) {
					blockSystem->addForce( &system.getForce( ltmd->forces[i].index ) );
				} else if( forcename == "Bond" ) {
					// Create a new harmonic bond force.
					// This only contains pairs of atoms which are in the same block.
					// I have to iterate through each bond from the old force, then
					// selectively add them to the new force based on this condition.
					HarmonicBondForce *hf = new HarmonicBondForce();
					const HarmonicBondForce *ohf = dynamic_cast<const HarmonicBondForce *>( &system.getForce( ltmd->forces[i].index ) );
					for( int i = 0; i < ohf->getNumBonds(); i++ ) {
						// For our system, add bonds between atoms in the same block
						int particle1, particle2;
						double length, k;
						ohf->getBondParameters( i, particle1, particle2, length, k );
						if( inSameBlock( particle1, particle2 ) ) {
							hf->addBond( particle1, particle2, length, k );
						}
					}
					blockSystem->addForce( hf );
				} else if( forcename == "Angle" ) {
					// Same thing with the angle force....
					HarmonicAngleForce *af = new HarmonicAngleForce();
					const HarmonicAngleForce *ahf = dynamic_cast<const HarmonicAngleForce *>( &system.getForce( ltmd->forces[i].index ) );
					for( int i = 0; i < ahf->getNumAngles(); i++ ) {
						// For our system, add bonds between atoms in the same block
						int particle1, particle2, particle3;
						double angle, k;
						ahf->getAngleParameters( i, particle1, particle2, particle3, angle, k );
						if( inSameBlock( particle1, particle2, particle3 ) ) {
							af->addAngle( particle1, particle2, particle3, angle, k );
						}
					}
					blockSystem->addForce( af );
				} else if( forcename == "Dihedral" ) {
					// And the dihedrals....
					PeriodicTorsionForce *ptf = new PeriodicTorsionForce();
					const PeriodicTorsionForce *optf = dynamic_cast<const PeriodicTorsionForce *>( &system.getForce( ltmd->forces[i].index ) );
					for( int i = 0; i < optf->getNumTorsions(); i++ ) {
						// For our system, add bonds between atoms in the same block
						int particle1, particle2, particle3, particle4, periodicity;
						double phase, k;
						optf->getTorsionParameters( i, particle1, particle2, particle3, particle4, periodicity, phase, k );
						if( inSameBlock( particle1, particle2, particle3, particle4 ) ) {
							ptf->addTorsion( particle1, particle2, particle3, particle4, periodicity, phase, k );
						}
					}
					blockSystem->addForce( ptf );
				} else if( forcename == "Improper" ) {
					// And the impropers....
					RBTorsionForce *rbtf = new RBTorsionForce();
					const RBTorsionForce *orbtf = dynamic_cast<const RBTorsionForce *>( &system.getForce( ltmd->forces[i].index ) );
					for( int i = 0; i < orbtf->getNumTorsions(); i++ ) {
						// For our system, add bonds between atoms in the same block
						int particle1, particle2, particle3, particle4;
						double c0, c1, c2, c3, c4, c5;
						orbtf->getTorsionParameters( i, particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5 );
						if( inSameBlock( particle1, particle2, particle3, particle4 ) ) {
							rbtf->addTorsion( particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5 );
						}
					}
					blockSystem->addForce( rbtf );
				} else if( forcename == "Nonbonded" ) {
					// This is a custom nonbonded pairwise force and
					// includes terms for both LJ and Coulomb.
					// Note that the step term will go to zero if block1 does not equal block 2,
					// and will be one otherwise.
					CustomBondForce *cbf = new CustomBondForce( "4*eps*((sigma/r)^12-(sigma/r)^6)+138.935456*q/r" );
					const NonbondedForce *nbf = dynamic_cast<const NonbondedForce *>( &system.getForce( ltmd->forces[i].index ) );
					NonbondedForce *nonbonded = new NonbondedForce();

					cbf->addPerBondParameter( "q" );
					cbf->addPerBondParameter( "sigma" );
					cbf->addPerBondParameter( "eps" );

					// store exceptions
					// exceptions[p1][p2] = params
					map<int, map<int, vector<double> > > exceptions;

					for( int i = 0; i < nbf->getNumExceptions(); i++ ) {
						int p1, p2;
						double q, sig, eps;
						nbf->getExceptionParameters( i, p1, p2, q, sig, eps );
						if( inSameBlock( p1, p2 ) ) {
							vector<double> params;
							params.push_back( q );
							params.push_back( sig );
							params.push_back( eps );
							if( exceptions.count( p1 ) == 0 ) {
								map<int, vector<double> > pair_exception;
								pair_exception[p2] = params;
								exceptions[p1] = pair_exception;
							} else {
								exceptions[p1][p2] = params;
							}
						}
					}

					// add particle params
					// TODO: iterate over block dimensions to reduce to O(b^2 N_b)
					for( int i = 0; i < nbf->getNumParticles() - 1; i++ ) {
						for( int j = i + 1; j < nbf->getNumParticles(); j++ ) {
							if( !inSameBlock( i, j ) ) {
								continue;
							}
							// we have an exception -- 1-4 modified interactions, etc.
							if( exceptions.count( i ) == 1 && exceptions[i].count( j ) == 1 ) {
								vector<double> params = exceptions[i][j];
								cbf->addBond( i, j, params );
							}
							// no exception, normal interaction
							else {
								vector<double> params;
								double q1, q2, eps1, eps2, sigma1, sigma2, q, eps, sigma;

								nbf->getParticleParameters( i, q1, sigma1, eps1 );
								nbf->getParticleParameters( j, q2, sigma2, eps2 );

								q = q1 * q2;
								sigma = 0.5 * ( sigma1 + sigma2 );
								eps = sqrt( eps1 * eps2 );

								params.push_back( q );
								params.push_back( sigma );
								params.push_back( eps );

								cbf->addBond( i, j, params );
							}
						}
					}

					blockSystem->addForce( cbf );
				} else {
					cout << "Unknown Force: " << forcename << endl;
				}
			}
			cout << "done." << endl;

			// Copy the positions.
			VerletIntegrator integ( 0.0 );
			if( blockContext ) {
				delete blockContext;
			}
			blockContext = new Context( *blockSystem, integ, Platform::getPlatformByName( "Reference" ) );
			bool isBlockDoublePrecision = blockContext->getPlatform().supportsDoublePrecision();
			vector<Vec3> blockPositions;
			for( int i = 0; i < numParticles; i++ ) {
				Vec3 atom( state.getPositions()[i][0], state.getPositions()[i][1], state.getPositions()[i][2] );
				blockPositions.push_back( atom );
			}

			blockContext->setPositions( blockPositions );
			/*********************************************************************/


			fstream perturb_forces;
			perturb_forces.open( "perturb_forces.txt", fstream::out );
			perturb_forces.precision( 10 );

			TNT::Array2D<double> h( n, n, 0.0 );
			largest_block_size *= 3; // degrees of freedom in the largest block
			vector<Vec3> initialBlockPositions( blockPositions );
			for( int i = 0; i < largest_block_size; i++ ) {
				vector<double> deltas( blocks.size() );
				// Perturb the ith degree of freedom in EACH block
				// Note: not all blocks will have i degrees, we have to check for this
				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= numParticles ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}

					double blockDelta = getDelta( blockPositions[atom_to_perturb][dof_to_perturb % 3], isBlockDoublePrecision, ltmd );
					deltas[j] = blockDelta;
					blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialBlockPositions[atom_to_perturb][dof_to_perturb % 3] - blockDelta;

				}

				blockContext->setPositions( blockPositions );
				vector<Vec3> forces1 = blockContext->getState( State::Forces ).getForces();



				// Now, do it again...
				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= numParticles ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}

					double blockDelta = deltas[j];
					blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialBlockPositions[atom_to_perturb][dof_to_perturb % 3] + blockDelta;


				}

				blockContext->setPositions( blockPositions );
				vector<Vec3> forces2 = blockContext->getState( State::Forces ).getForces();

				// write out forces
				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= numParticles ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}

					int start_dof = 3 * blocks[j];
					int end_dof;
					if( j == blocks.size() - 1 ) {
						end_dof = 3 * numParticles;
					} else {
						end_dof = 3 * blocks[j + 1];
					}

					for( int k = start_dof; k < end_dof; k++ ) {
						perturb_forces << k << " " << dof_to_perturb << " " << forces2[k / 3][k % 3] << endl;
					}

				}


				// revert block positions
				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= numParticles ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}

					blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialBlockPositions[atom_to_perturb][dof_to_perturb % 3];

				}

				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= numParticles ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}

					int col = dof_to_perturb; //(atom_to_perturb*3)+(dof_to_perturb % 3);
					int row = 0;

					int start_dof = 3 * blocks[j];
					int end_dof;
					if( j == blocks.size() - 1 ) {
						end_dof = 3 * numParticles;
					} else {
						end_dof = 3 * blocks[j + 1];
					}

					double blockDelta = deltas[j];
					for( int k = start_dof; k < end_dof; k++ ) {
						double blockscale = 1.0 / ( 2 * blockDelta * sqrt( system.getParticleMass( atom_to_perturb ) * system.getParticleMass( k / 3 ) ) );
						h[k][col] = ( forces1[k / 3][k % 3] - forces2[k / 3][k % 3] ) * blockscale;
					}
				}

			}
			perturb_forces.close();

			gettimeofday( &tp_hess, NULL );
			cout << "Time to compute hessian: " << ( tp_hess.tv_sec - tp_begin.tv_sec ) << endl;

			fstream block_hessian;
			block_hessian.open( "block_hessian.txt", fstream::out );
			block_hessian.precision( 10 );
			for( int i = 0; i < 3 * numParticles; i++ ) {
				for( int j = 0; j < 3 * numParticles; j++ ) {
					if( h[i][j] != 0.0 ) {
						block_hessian << i << " " << j << " " << h[i][j] << endl;
					}
				}
			}
			block_hessian.close();


			// Make sure it is exactly symmetric.
			for( int i = 0; i < n; i++ ) {
				for( int j = 0; j < i; j++ ) {
					double avg = 0.5f * ( h[i][j] + h[j][i] );
					h[i][j] = avg;
				}
			}




			// Diagonalize each block Hessian, get Eigenvectors
			// Note: The eigenvalues will be placed in one large array, because
			//       we must sort them to get k
			//vector<double> Di;
			const int cdof = 6;
			TNT::Array1D<double> block_eigval( n, 0.0 );
			TNT::Array2D<double> block_eigvec( n, n, 0.0 );
			int total_surviving_eigvec = 0;
			fstream all_eigs;
			all_eigs.open( "all_eigs.txt", fstream::out );
			all_eigs.precision( 10 );
			for( int i = 0; i < blocks.size(); i++ ) {
				cout << "Diagonalizing block: " << i << endl;
				// 1. Determine the starting and ending index for the block
				//    This means that the upper left corner of the block will be at (startatom, startatom)
				//    And the lower right corner will be at (endatom, endatom)
				int startatom = 3 * blocks[i];
				int endatom;
				if( i == blocks.size() - 1 ) {
					endatom = 3 * numParticles - 1;
				} else {
					endatom = 3 * blocks[i + 1] - 1;
				}

				const int size = endatom - startatom + 1;

				// 2. Get the block Hessian Hii
				//    Right now I'm just doing a copy from the big Hessian
				//    There's probably a more efficient way but for now I just want things to work..
				TNT::Array2D<double> h_tilde( size, size, 0.0 );
				for( int j = startatom; j <= endatom; j++ ) {
					for( int k = startatom; k <= endatom; k++ ) {
						h_tilde[k - startatom][j - startatom] = h[k][j];
					}
				}

				// 3. Diagonalize the block Hessian only, and get eigenvectors
				TNT::Array1D<double> di( size, 0.0 );
				TNT::Array2D<double> Qi( size, size, 0.0 );
				findEigenvaluesJama( h_tilde, di, Qi );
				//findEigenvaluesLapack(h_tilde, di, Qi);

				// sort eigenvalues by absolute magnitude
				vector<pair<double, int> > sortedEvalPairs( size );
				for( int j = 0; j < size; j++ ) {
					sortedEvalPairs.at( j ) = make_pair( fabs( di[j] ), j );
				}
				sort( sortedEvalPairs.begin(), sortedEvalPairs.end() );

				// find geometric dof
				TNT::Array2D<double> Qi_gdof( size, size, 0.0 );

				Vec3 pos_center( 0.0, 0.0, 0.0 );
				double totalmass = 0.0;

				for( int j = startatom; j <= endatom; j += 3 ) {
					double mass = system.getParticleMass( j / 3 );
					pos_center += positions[j / 3] * mass;
					totalmass += mass;
				}

				double norm = sqrt( totalmass );

				// actual center
				pos_center *= 1.0 / totalmass;

				// create geometric dof vectors
				// iterating over rows and filling in values for 6 vectors as we go
				for( int j = 0; j < size; j += 3 ) {
					double atom_index = ( startatom + j ) / 3;
					double mass = system.getParticleMass( atom_index );
					double factor = sqrt( mass ) / norm;

					// translational
					Qi_gdof[j][0]   = factor;
					Qi_gdof[j + 1][1] = factor;
					Qi_gdof[j + 2][2] = factor;

					// rotational
					// cross product of rotation axis and vector to center of molecule
					// x-axis (b1=1) ja3-ka2
					// y-axis (b2=1) ka1-ia3
					// z-axis (b3=1) ia2-ja1
					Vec3 diff = positions[atom_index] - pos_center;
					// x
					Qi_gdof[j + 1][3] =  diff[2] * factor;
					Qi_gdof[j + 2][3] = -diff[1] * factor;

					// y
					Qi_gdof[j][4]   = -diff[2] * factor;
					Qi_gdof[j + 2][4] =  diff[0] * factor;

					// z
					Qi_gdof[j][5]   =  diff[1] * factor;
					Qi_gdof[j + 1][5] = -diff[0] * factor;
				}

				fstream gdof_out;
				gdof_out.open( "gdof_vec.txt", fstream::out | fstream::app );
				gdof_out.precision( 10 );
				// iterate over rows
				for( int j = 0; j < size; j ++ ) {
					for( int k = 0; k < cdof; k++ ) {
						int mycol = i * cdof + k;
						gdof_out << startatom + j << " " << mycol << " " << Qi_gdof[j][k] << endl;
					}
				}
				gdof_out.close();

				// normalize first rotational vector
				double rotnorm = 0.0;
				for( int j = 0; j < size; j++ ) {
					rotnorm += Qi_gdof[j][3] * Qi_gdof[j][3];
				}

				rotnorm = 1.0 / sqrt( rotnorm );

				for( int j = 0; j < size; j++ ) {
					Qi_gdof[j][3] = Qi_gdof[j][3] * rotnorm;
				}

				// orthogonalize rotational vectors 2 and 3
				for( int j = 4; j < cdof; j++ ) { // <-- vector we're orthogonalizing
					for( int k = 3; k < j; k++ ) { // <-- vectors we're orthognalizing against
						double dot_prod = 0.0;
						for( int l = 0; l < size; l++ ) {
							dot_prod += Qi_gdof[l][k] * Qi_gdof[l][j];
						}
						for( int l = 0; l < size; l++ ) {
							Qi_gdof[l][j] = Qi_gdof[l][j] - Qi_gdof[l][k] * dot_prod;
						}
					}

					// normalize residual vector
					double rotnorm = 0.0;
					for( int l = 0; l < size; l++ ) {
						rotnorm += Qi_gdof[l][j] * Qi_gdof[l][j];
						//rotnorm += Qi_gdof.at(l)->at(j) * Qi_gdof.at(l)->at(j);
					}

					rotnorm = 1.0 / sqrt( rotnorm );

					for( int l = 0; l < size; l++ ) {
						Qi_gdof[l][j] = Qi_gdof[l][j] * rotnorm;
					}
				}

				fstream gdof_out_orth;
				gdof_out_orth.open( "gdof_vec_orth.txt", fstream::out | fstream::app );
				gdof_out_orth.precision( 10 );
				// iterate over rows
				for( int j = 0; j < size; j ++ ) {
					for( int k = 0; k < cdof; k++ ) {
						int mycol = i * cdof + k;
						gdof_out_orth << startatom + j << " " << mycol << " " << Qi_gdof[j][k] << endl;
					}
				}
				gdof_out_orth.close();


				// orthogonalize original eigenvectors against gdof
				// number of evec that survive orthogonalization
				int curr_evec = cdof;
				for( int j = 0; j < size; j++ ) { // <-- vector we're orthogonalizing
					// to match ProtoMol we only include size instead of size + cdof vectors
					// Note: for every vector that is skipped due to a low norm,
					// we add an additional vector to replace it, so we could actually
					// use all size original eigenvectors
					if( curr_evec == size ) {
						break;
					}

					// orthogonalize original eigenvectors in order from smallest magnitude
					// eigenvalue to biggest
					int col = sortedEvalPairs.at( j ).second;

					// copy original vector to Qi_gdof -- updated in place
					for( int l = 0; l < size; l++ ) {
						Qi_gdof[l][curr_evec] = Qi[l][col];
					}

					// get dot products with previous vectors
					for( int k = 0; k < curr_evec; k++ ) { // <-- vector orthog against
						// dot product between original vector and previously
						// orthogonalized vectors
						double dot_prod = 0.0;
						for( int l = 0; l < size; l++ ) {
							dot_prod += Qi_gdof[l][k] * Qi[l][col];
						}

						// subtract from current vector -- update in place
						for( int l = 0; l < size; l++ ) {
							Qi_gdof[l][curr_evec] = Qi_gdof[l][curr_evec] - Qi_gdof[l][k] * dot_prod;
						}
					}

					//normalize residual vector
					double norm = 0.0;
					for( int l = 0; l < size; l++ ) {
						norm += Qi_gdof[l][curr_evec] * Qi_gdof[l][curr_evec];
					}

					// if norm less than 1/20th of original
					// continue on to next vector
					// we don't update curr_evec so this vector
					// will be overwritten
					if( norm < 0.05 ) {
						cout << "skipping vec" << endl;
						continue;
					}

					// scale vector
					norm = sqrt( norm );
					for( int l = 0; l < size; l++ ) {
						Qi_gdof[l][curr_evec] = Qi_gdof[l][curr_evec] / norm;
					}

					curr_evec++;
				}


				fstream block_out;
				stringstream ss;
				ss << "block_" << i << ".txt";
				block_out.open( ss.str().c_str(), fstream::out );
				block_out.precision( 10 );
				// iterate over rows
				for( int j = 0; j < size; j ++ ) {
					for( int k = 0; k < curr_evec; k++ ) {
						if( Qi_gdof[j][k] != 0.0 ) {
							block_out << j << " " << k << " " << Qi_gdof[j][k] << endl;
						}
					}
				}
				block_out.close();


				cout << "curr evec " << curr_evec << endl;
				cout << "size " << size << endl;

				// 4. Copy eigenpairs to big array
				//    This is necessary because we have to sort them, and determine
				//    the cutoff eigenvalue for everybody.
				// we assume curr_evec <= size
				for( int j = 0; j < curr_evec; j++ ) {
					int col = sortedEvalPairs.at( j ).second;
					block_eigval[total_surviving_eigvec] = di[col];

					// orthogonalized eigenvectors already sorted by eigenvalue
					for( int k = 0; k < size; k++ ) {
						block_eigvec[startatom + k][total_surviving_eigvec] = Qi_gdof[k][j];
					}
					total_surviving_eigvec++;
				}
			}

			fstream block_out;
			stringstream ss;
			ss << "all_block_vec" << ".txt";
			block_out.open( ss.str().c_str(), fstream::out );
			block_out.precision( 10 );
			// iterate over rows
			for( int j = 0; j < n; j ++ ) {
				for( int k = 0; k < total_surviving_eigvec; k++ ) {
					block_out << j << " " << k << " " << block_eigvec[j][k] << endl;
				}
			}
			block_out.close();



			gettimeofday( &tp_diag, NULL );
			cout << "Time to diagonalize block hessian: " << ( tp_diag.tv_sec - tp_hess.tv_sec ) << endl;


			//***********************************************************
			// This section here is only to find the cuttoff eigenvalue.
			// First sort the eigenvectors by the absolute value of the eigenvalue.

			cout << "total surviving eigs " << total_surviving_eigvec << endl;

			// sort all eigenvalues by absolute magnitude to determine cutoff
			vector<pair<double, int> > sortedEvalues( total_surviving_eigvec );
			for( int i = 0; i < total_surviving_eigvec; i++ ) {
				sortedEvalues[i] = make_pair( fabs( block_eigval[i] ), i );
			}
			sort( sortedEvalues.begin(), sortedEvalues.end() );

			int max_eigs = ltmd->bdof * blocks.size();
			double cutEigen = sortedEvalues[max_eigs].first;  // This is the cutoff eigenvalue
			cout << "cutoff " << cutEigen << endl;

			// get cols of all eigenvalues under cutoff
			vector<int> selectedEigsCols;
			for( int i = 0; i < total_surviving_eigvec; i++ ) {
				if( fabs( block_eigval[i] ) < cutEigen ) {
					selectedEigsCols.push_back( i );
				}
			}

			cout << "selected " << selectedEigsCols.size() << " eigs" << endl;
			cout << "max_eigs " << max_eigs << endl;

			// we may select fewer eigs if there are duplicate eigenvalues
			const int m = selectedEigsCols.size();

			fstream eigs_output;
			eigs_output.open( "block_eigs.txt", fstream::out );
			eigs_output.precision( 10 );
			cout << "opened the file" << endl;
			for( int i = 0; i < selectedEigsCols.size(); i++ ) {
				int eig_col = selectedEigsCols.at( i );

				for( int j = 0; j < n; j++ ) {
					eigs_output << j << " " << i << " " << block_eigvec[j][eig_col] << endl;
				}
			}
			eigs_output.close();

			cout << "output selected" << endl;

			// Inefficient, needs to improve.
			// Basically, just setting up E and E^T by
			// copying values from bigE.
			// Again, right now I'm only worried about
			// correctness plus this time will be marginal compared to
			// diagonalization.
			cout << "M: " << m << endl;
			TNT::Array2D<double> E( n, m, 0.0 );
			TNT::Array2D<double> E_transpose( m, n, 0.0 );
			//TNT::Array2D<float> EPS(n, m);
			//TNT::Array2D<float> EPS_transpose(m, n);
			for( int i = 0; i < m; i++ ) {
				int eig_col = selectedEigsCols[i];
				for( int j = 0; j < n; j++ ) {
					E_transpose[i][j] = block_eigvec[j][eig_col];
					E[j][i] = block_eigvec[j][eig_col];
				}
			}
			gettimeofday( &tp_e, NULL );
			cout << "Time to compute E: " << ( tp_e.tv_sec - tp_diag.tv_sec ) << endl;


			//*****************************************************************
			// Compute S, which is equal to E^T * H * E.
			// Using the matmult function of Jama.
			TNT::Array2D<double> S( m, m, 0.0 );

			// Compute eps.
			const double eps = ltmd->delta;

			// Make a temp copy of positions.
			vector<Vec3> tmppos( positions );
			// Loop over i.
			for( unsigned int k = 0; k < m; k++ ) {
				// Perturb positions.
				int pos = 0;
				// forward perturbations
				for( unsigned int i = 0; i < numParticles; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j] + eps * E[3 * i + j][k] / sqrt( system.getParticleMass( i ) );
						pos++;
					}
				}
				context.setPositions( tmppos );

				// Calculate F(xi).
				vector<Vec3> forces_forward = context.getState( State::Forces ).getForces();

				// backward perturbations
				for( unsigned int i = 0; i < numParticles; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j] - eps * E[3 * i + j][k] / sqrt( system.getParticleMass( i ) );
					}
				}
				context.setPositions( tmppos );

				// Calculate forces
				vector<Vec3> forces_backward = context.getState( State::Forces ).getForces();

				TNT::Array2D<double> Force_diff( n, 1, 0.0 );
				for( int i = 0; i < n; i++ ) {
					const double scaleFactor = sqrt( system.getParticleMass( i / 3 ) ) * 2.0 * eps;
					Force_diff[i][0] = ( forces_forward[i / 3][i % 3] - forces_backward[i / 3][i % 3] ) / scaleFactor;
				}

				TNT::Array2D<double> Si( m, 1, 0.0 );
				//matMultLapack(EPS_transpose,Force_diff,Si);
				Si = matmult( E_transpose, Force_diff );

				// Copy to S.
				for( int i = 0; i < m; i++ ) {
					S[i][k] = Si[i][0];
				}

				// restore positions
				for( unsigned int i = 0; i < numParticles; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j];
					}
				}
			}

			// *****************************************************************
			// restore unperturbed positions
			context.setPositions( positions );

			// make S symmetric
			for( unsigned int i = 0; i < S.dim1(); i++ ) {
				for( unsigned int j = 0; j < S.dim2(); j++ ) {
					double avg = 0.5f * ( S[i][j] + S[j][i] );
					S[i][j] = avg;
					S[j][i] = avg;
				}
			}

			fstream s_out;
			s_out.open( "s.txt", fstream::out );
			s_out.precision( 10 );
			for( int i = 0; i < m; i++ ) {
				for( int j = 0; j < m; j++ ) {
					s_out << j << " " << i << " " << S[j][i] << endl;
				}
			}
			s_out.close();


			gettimeofday( &tp_s, NULL );
			cout << "Time to compute S: " << ( tp_s.tv_sec - tp_e.tv_sec ) << endl;
			// Diagonalizing S by finding eigenvalues and eigenvectors...
			TNT::Array1D<double> dS( m, 0.0 );
			TNT::Array2D<double> q( m, m, 0.0 );
			findEigenvaluesJama( S, dS, q );
			//findEigenvaluesLapack(S, dS, q);


			fstream s_eig_out;
			s_eig_out.open( "s_eig_out.txt", fstream::out );
			s_eig_out.precision( 10 );
			for( int i = 0; i < S.dim1(); i++ ) {
				for( int j = 0; j < S.dim1(); j++ ) {
					s_eig_out << j << " " << i << " " << q[j][i] << endl;
				}
			}

			// Sort by ABSOLUTE VALUE of eigenvalues.
			sortedEvalues.clear();
			sortedEvalues.resize( dS.dim() );
			for( int i = 0; i < dS.dim(); i++ ) {
				sortedEvalues[i] = make_pair( fabs( dS[i] ), i );
			}
			sort( sortedEvalues.begin(), sortedEvalues.end() );

			TNT::Array2D<double> Q( q.dim2(), q.dim1(), 0.0 );
			for( int i = 0; i < sortedEvalues.size(); i++ )
				for( int j = 0; j < q.dim2(); j++ ) {
					Q[j][i] = q[j][sortedEvalues[i].second];
				}
			maxEigenvalue = sortedEvalues[dS.dim() - 1].first;
			gettimeofday( &tp_q, NULL );
			cout << "Time to compute Q: " << ( tp_q.tv_sec - tp_s.tv_sec ) << endl;


			// Compute U, set of approximate eigenvectors.
			// U = E*Q.
			TNT::Array2D<double> U = matmult( E, Q ); //E*Q;
			gettimeofday( &tp_u, NULL );
			cout << "Time to compute U: " << ( tp_u.tv_sec - tp_q.tv_sec ) << endl;

			// Record the eigenvectors.
			// These will be placed in a file eigenvectors.txt
			ofstream outfile( "eigenvectors.txt", ios::out );
			eigenvectors.resize( numVectors, vector<Vec3>( numParticles ) );
			for( int i = 0; i < numVectors; i++ ) {
				for( int j = 0; j < numParticles; j++ ) {
					eigenvectors[i][j] = Vec3( U[3 * j][i], U[3 * j + 1][i], U[3 * j + 2][i] );
					outfile << 3 * j << " " << i << " " << U[3 * j][i] << endl;
					outfile << 3 * j + 1 << " " << i << " " << U[3 * j + 1][i] << endl;
					outfile << 3 * j + 2 << " " << i << " " << U[3 * j + 2][i] << endl;
				}
			}


			gettimeofday( &tp_end, NULL );
			cout << "Overall diagonalization time in seconds: " << ( tp_end.tv_sec - tp_begin.tv_sec ) << endl;
		}

		void Analysis::computeEigenvectorsRestricting( ContextImpl &contextImpl, int numVectors ) {
			Context &context = contextImpl.getOwner();
			bool isDoublePrecision = context.getPlatform().supportsDoublePrecision();
			State state = context.getState( State::Positions | State::Forces );
			vector<Vec3> positions = state.getPositions();
			vector<Vec3> forces = state.getForces();
			System &system = context.getSystem();
			int numParticles = positions.size();
			int n = 3 * numParticles;

			// Construct the mass weighted Hessian.

			TNT::Array2D<float> h( n, n );
			for( int i = 0; i < numParticles; i++ ) {
				Vec3 pos = positions[i];
				for( int j = 0; j < 3; j++ ) {
					double delta = getDelta( positions[i][j], isDoublePrecision, NULL );
					positions[i][j] = pos[j] + delta;
					context.setPositions( positions );
					vector<Vec3> forces2 = context.getState( State::Forces ).getForces();
					positions[i][j] = pos[j];
					int col = i * 3 + j;
					int row = 0;
					for( int k = 0; k < numParticles; k++ ) {
						double scale = 1.0 / ( delta * sqrt( system.getParticleMass( i ) * system.getParticleMass( k ) ) );
						h[row++][col] = ( forces[k][0] - forces2[k][0] ) * scale;
						h[row++][col] = ( forces[k][1] - forces2[k][1] ) * scale;
						h[row++][col] = ( forces[k][2] - forces2[k][2] ) * scale;
					}
				}
			}

			// Make sure it is exactly symmetric.

			for( int i = 0; i < n; i++ ) {
				for( int j = 0; j < i; j++ ) {
					float avg = 0.5f * ( h[i][j] + h[j][i] );
					h[i][j] = avg;
					h[j][i] = avg;
				}
			}

			// Find a projection matrix to the smaller subspace.

			const vector<vector<int> >& molecules = contextImpl.getMolecules();
			buildTree( contextImpl );
			int m = 6 * molecules.size() + bonds.size();
			projection.resize( m, vector<double>( n, 0.0 ) );
			for( int i = 0; i < ( int ) molecules.size(); i++ ) {
				// Find the center of the molecule.

				const vector<int>& mol = molecules[i];
				Vec3 center;
				for( int j = 0; j < ( int ) mol.size(); j++ ) {
					center += positions[mol[j]];
				}
				center *= 1.0 / mol.size();

				// Now loop over particles.

				for( int j = 0; j < ( int ) mol.size(); j++ ) {
					// Fill in the projection matrix.

					int particle = mol[j];
					projection[6 * i][3 * particle] = 1.0;
					projection[6 * i + 1][3 * particle + 1] = 1.0;
					projection[6 * i + 2][3 * particle + 2] = 1.0;
					Vec3 pos = positions[particle];
					projection[6 * i + 3][3 * particle + 1] = -( pos[2] - center[2] );
					projection[6 * i + 3][3 * particle + 2] = ( pos[1] - center[1] );
					projection[6 * i + 4][3 * particle + 0] = ( pos[2] - center[2] );
					projection[6 * i + 4][3 * particle + 2] = -( pos[0] - center[0] );
					projection[6 * i + 5][3 * particle + 0] = -( pos[1] - center[1] );
					projection[6 * i + 5][3 * particle + 1] = ( pos[0] - center[0] );
				}
			}
			for( int i = 0; i < ( int ) bonds.size(); i++ ) {
				Vec3 base = positions[bonds[i].first];
				Vec3 dir = positions[bonds[i].second] - base;
				dir *= 1.0 / sqrt( dir.dot( dir ) );
				vector<int> children;
				findChildren( *particleNodes[bonds[i].second], children );
				int row = 6 * molecules.size() + i;
				for( int j = 0; j < ( int ) children.size(); j++ ) {
					int particle = children[j];
					Vec3 delta = dir.cross( positions[particle] - base );
					projection[row][3 * particle] = delta[0];
					projection[row][3 * particle + 1] = delta[1];
					projection[row][3 * particle + 2] = delta[2];
				}
			}
			for( int i = 0; i < m; i++ ) {
				for( int j = 0; j < numParticles; j++ ) {
					double scale = sqrt( system.getParticleMass( j ) );
					projection[i][3 * j] *= scale;
					projection[i][3 * j + 1] *= scale;
					projection[i][3 * j + 2] *= scale;
				}
			}
			for( int i = 0; i < m; i++ ) {
				// Make this vector orthogonal to all previous ones.

				for( int j = 0; j < i; j++ ) {
					double dot = 0.0;
					for( int k = 0; k < n; k++ ) {
						dot += projection[i][k] * projection[j][k];
					}
					for( int k = 0; k < n; k++ ) {
						projection[i][k] -= dot * projection[j][k];
					}
				}

				// Normalize it.

				double sum = 0.0;
				for( int j = 0; j < n; j++ ) {
					sum += projection[i][j] * projection[i][j];
				}
				double scale = 1.0 / sqrt( sum );
				for( int j = 0; j < n; j++ ) {
					projection[i][j] *= scale;
				}
			}

			// Multiply by the projection matrix to get an m by m Hessian.

			vector<vector<double> > h2( m, vector<double>( n ) );
			for( int i = 0; i < m; i++ )
				for( int j = 0; j < n; j++ ) {
					double sum = 0.0;
					for( int k = 0; k < n; k++ ) {
						sum += projection[i][k] * h[k][j];
					}
					h2[i][j] = sum;
				}
			TNT::Array2D<float> s( m, m );
			for( int i = 0; i < m; i++ )
				for( int j = 0; j < m; j++ ) {
					double sum = 0.0;
					for( int k = 0; k < n; k++ ) {
						sum += projection[i][k] * h2[j][k];
					}
					s[i][j] = sum;
				}


			// Sort the eigenvectors by the absolute value of the eigenvalue.

			JAMA::Eigenvalue<float> decomp( s );
			TNT::Array1D<float> d;
			decomp.getRealEigenvalues( d );
			vector<pair<float, int> > sortedEigenvalues( m );
			for( int i = 0; i < m; i++ ) {
				sortedEigenvalues[i] = make_pair( fabs( d[i] ), i );
			}
			sort( sortedEigenvalues.begin(), sortedEigenvalues.end() );
			maxEigenvalue = sortedEigenvalues[m - 1].first;

			// Record the eigenvectors.

			TNT::Array2D<float> eigen;
			decomp.getV( eigen );
			eigenvectors.resize( numVectors, vector<Vec3>( numParticles ) );
			for( int i = 0; i < numVectors; i++ ) {
				int index = sortedEigenvalues[i].second;
				for( int j = 0; j < n; j += 3 ) {
					Vec3 sum;
					for( int k = 0; k < m; k++ ) {
						sum[0] += projection[k][j] * eigen[k][index];
						sum[1] += projection[k][j + 1] * eigen[k][index];
						sum[2] += projection[k][j + 2] * eigen[k][index];
					}
					eigenvectors[i][j / 3] = sum;
				}
			}
		}

		void Analysis::computeEigenvectorsDihedral( ContextImpl &contextImpl, int numVectors ) {
			Context &context = contextImpl.getOwner();
			State state = context.getState( State::Positions | State::Forces );
			vector<Vec3> positions = state.getPositions();
			vector<Vec3> forces = state.getForces();
			int numParticles = positions.size();
			System &system = context.getSystem();
			const vector<vector<int> >& molecules = contextImpl.getMolecules();
			buildTree( contextImpl );
			int n = 3 * numParticles;
			int m = 6 * molecules.size() + bonds.size();
			projection.resize( m, vector<double>( n, 0.0 ) );

			// First compute derivatives with respect to global translations and rotations.

			for( int i = 0; i < ( int ) molecules.size(); i++ ) {
				// Find the center of the molecule.

				const vector<int>& mol = molecules[i];
				Vec3 center;
				for( int j = 0; j < ( int ) mol.size(); j++ ) {
					center += positions[mol[j]];
				}
				center *= 1.0 / mol.size();

				// Now loop over particles.

				for( int j = 0; j < ( int ) mol.size(); j++ ) {
					// Fill in the projection matrix.

					int particle = mol[j];
					projection[6 * i][3 * particle] = 1.0;
					projection[6 * i + 1][3 * particle + 1] = 1.0;
					projection[6 * i + 2][3 * particle + 2] = 1.0;
					Vec3 pos = positions[particle];
					projection[6 * i + 3][3 * particle + 1] = -( pos[2] - center[2] );
					projection[6 * i + 3][3 * particle + 2] = ( pos[1] - center[1] );
					projection[6 * i + 4][3 * particle + 0] = ( pos[2] - center[2] );
					projection[6 * i + 4][3 * particle + 2] = -( pos[0] - center[0] );
					projection[6 * i + 5][3 * particle + 0] = -( pos[1] - center[1] );
					projection[6 * i + 5][3 * particle + 1] = ( pos[0] - center[0] );
				}
			}

			// Compute derivatives with respect to dihedrals.

			for( int i = 0; i < ( int ) bonds.size(); i++ ) {
				Vec3 base = positions[bonds[i].first];
				Vec3 dir = positions[bonds[i].second] - base;
				dir *= 1.0 / sqrt( dir.dot( dir ) );
				vector<int> children;
				findChildren( *particleNodes[bonds[i].second], children );
				int row = 6 * molecules.size() + i;
				for( int j = 0; j < ( int ) children.size(); j++ ) {
					int particle = children[j];
					Vec3 delta = dir.cross( positions[particle] - base );
					projection[row][3 * particle] = delta[0];
					projection[row][3 * particle + 1] = delta[1];
					projection[row][3 * particle + 2] = delta[2];
				}
			}
			for( int i = 0; i < m; i++ ) {
				for( int j = 0; j < numParticles; j++ ) {
					double scale = sqrt( system.getParticleMass( j ) );
					projection[i][3 * j] *= scale;
					projection[i][3 * j + 1] *= scale;
					projection[i][3 * j + 2] *= scale;
				}
			}
			for( int i = 0; i < m; i++ ) {
				// Make this vector orthogonal to all previous ones.

				for( int j = 0; j < i; j++ ) {
					double dot = 0.0;
					for( int k = 0; k < n; k++ ) {
						dot += projection[i][k] * projection[j][k];
					}
					for( int k = 0; k < n; k++ ) {
						projection[i][k] -= dot * projection[j][k];
					}
				}

				// Normalize it.

				double sum = 0.0;
				for( int j = 0; j < n; j++ ) {
					sum += projection[i][j] * projection[i][j];
				}
				double scale = 1.0 / sqrt( sum );
				for( int j = 0; j < n; j++ ) {
					projection[i][j] *= scale;
				}
			}

			// Construct an m by n "Hessian like" matrix.

			vector<vector<double> > h( m, vector<double>( n ) );
			vector<Vec3> positions2( numParticles );
			for( int i = 0; i < m; i++ ) {
				double delta = sqrt( 1e-7 );
				for( int j = 0; j < numParticles; j++ )
					for( int k = 0; k < 3; k++ ) {
						positions2[j][k] = positions[j][k] + delta * projection[i][3 * j + k];    ///sqrt(system.getParticleMass(j));
					}
				context.setPositions( positions2 );
				vector<Vec3> forces2 = context.getState( State::Forces ).getForces();
				for( int j = 0; j < numParticles; j++ ) {
					double scale = 1.0 / delta;
					h[i][3 * j] = ( forces[j][0] - forces2[j][0] ) * scale;
					h[i][3 * j + 1] = ( forces[j][1] - forces2[j][1] ) * scale;
					h[i][3 * j + 2] = ( forces[j][2] - forces2[j][2] ) * scale;
				}
			}

			// Multiply by the projection matrix to get an m by m Hessian.

			TNT::Array2D<float> s( m, m );
			for( int i = 0; i < m; i++ )
				for( int j = 0; j < m; j++ ) {
					double sum = 0.0;
					for( int k = 0; k < n; k++ ) {
						sum += projection[i][k] * h[j][k];
					}
					s[i][j] = sum;
				}

			// Make sure it is exactly symmetric.

			for( int i = 0; i < m; i++ ) {
				for( int j = 0; j < i; j++ ) {
					float avg = 0.5f * ( s[i][j] + s[j][i] );
					s[i][j] = avg;
					s[j][i] = avg;
				}
			}

			// Sort the eigenvectors by the absolute value of the eigenvalue.

			JAMA::Eigenvalue<float> decomp( s );
			TNT::Array1D<float> d;
			decomp.getRealEigenvalues( d );
			vector<pair<float, int> > sortedEigenvalues( m );
			for( int i = 0; i < m; i++ ) {
				sortedEigenvalues[i] = make_pair( fabs( d[i] ), i );
			}
			sort( sortedEigenvalues.begin(), sortedEigenvalues.end() );
			maxEigenvalue = sortedEigenvalues[m - 1].first;

			// Record the eigenvectors.

			TNT::Array2D<float> eigen;
			decomp.getV( eigen );
			eigenvectors.resize( numVectors, vector<Vec3>( numParticles ) );
			for( int i = 0; i < numVectors; i++ ) {
				int index = sortedEigenvalues[i].second;
				for( int j = 0; j < n; j += 3 ) {
					Vec3 sum;
					for( int k = 0; k < m; k++ ) {
						sum[0] += projection[k][j] * eigen[k][index];
						sum[1] += projection[k][j + 1] * eigen[k][index];
						sum[2] += projection[k][j + 2] * eigen[k][index];
					}
					eigenvectors[i][j / 3] = sum;
				}
			}
		}

		double Analysis::getDelta( double value, bool isDoublePrecision, Parameters *ltmd ) {
			double delta = sqrt( ltmd->delta ) * max( fabs( value ), 0.1 );
			//double delta = sqrt(isDoublePrecision ? 1e-16 : 1e-7)*max(fabs(value), 0.1);
			volatile double temp = value + delta;
			delta = temp - value;
			return ltmd->delta;
		}

		void Analysis::buildTree( ContextImpl &context ) {
			System &system = context.getSystem();
			int numParticles = system.getNumParticles();
			int numConstraints = system.getNumConstraints();
			vector<pair<int, int> > allBonds( numConstraints );
			for( int i = 0; i < numConstraints; i++ ) {
				double dist;
				system.getConstraintParameters( i, allBonds[i].first, allBonds[i].second, dist );
			}
			for( int i = 0; i < ( int ) context.getForceImpls().size(); i++ ) {
				const ForceImpl &force = *context.getForceImpls()[i];
				const vector<pair<int, int> >& forceBonds = force.getBondedParticles();
				for( int j = 0; j < ( int ) forceBonds.size(); j++ ) {
					allBonds.push_back( forceBonds[j] );
				}
			}
			particleBonds.resize( numParticles );
			for( int i = 0; i < ( int ) allBonds.size(); i++ ) {
				particleBonds[allBonds[i].first].push_back( allBonds[i].second );
				particleBonds[allBonds[i].second].push_back( allBonds[i].first );
			}



			vector<bool> processed( numParticles, false );
			for( int i = 0; i < numParticles; i++ )
				if( !processed[i] ) {
					treeRoots.push_back( TreeNode( i ) );
					processed[i] = true;
					processTreeNode( treeRoots.back(), processed, true );
				}
			for( int i = 0; i < ( int ) treeRoots.size(); i++ ) {
				recordParticleNodes( treeRoots[i] );
			}
		}

		void Analysis::processTreeNode( TreeNode &node, vector<bool>& processed, bool isRootNode ) {
			vector<int>& bonded = particleBonds[node.particle];
			for( int i = 0; i < ( int ) bonded.size(); i++ )
				if( !processed[bonded[i]] ) {
					node.children.push_back( TreeNode( bonded[i] ) );
					processed[bonded[i]] = true;
					processTreeNode( node.children.back(), processed, false );
					node.totalChildren += node.children.back().totalChildren + 1;
					if( node.children.back().totalChildren > 0 && !isRootNode ) {
						bonds.push_back( make_pair( node.particle, bonded[i] ) );
					}
				}
		}

		void Analysis::recordParticleNodes( TreeNode &node ) {
			particleNodes[node.particle] = &node;
			for( int i = 0; i < ( int ) node.children.size(); i++ ) {
				recordParticleNodes( node.children[i] );
			}
		}

		void Analysis::findChildren( const TreeNode &node, std::vector<int>& children ) const {
			for( int i = 0; i < ( int ) node.children.size(); i++ ) {
				children.push_back( node.children[i].particle );
				findChildren( node.children[i], children );
			}
		}
	}
}
