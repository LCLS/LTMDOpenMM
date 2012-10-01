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
#include <algorithm>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "OpenMM.h"
#include "LTMD/Math.h"
#include "LTMD/Analysis.h"
#include "LTMD/Integrator.h"

#include "tnt_array2d_utils.h"

namespace OpenMM {
	namespace LTMD {
		const unsigned int ConservedDegreesOfFreedom = 6;
		
		// Function Implementations
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
		
		bool sort_func( const EigenvalueColumn& a, const EigenvalueColumn& b ) {
			if( std::fabs( a.first - b.first ) < 1e-8 ) {
				if( a.second <= b.second ) return true;
			}else{
				if( a.first < b.first ) return true;
			}
			
			return false;
		}
		
		std::vector<EigenvalueColumn> Analysis::SortEigenvalues( const EigenvalueArray& values ) {
			std::vector<EigenvalueColumn> retVal;
			
			// Create Array
			retVal.reserve( values.size() );
			for( unsigned int i = 0; i < values.size(); i++ ){
				retVal.push_back( std::make_pair( std::fabs( values[i] ), i ) );
			}
			
			// Sort Data
			std::sort( retVal.begin(), retVal.end(), sort_func );
			
			return retVal;
		}
		
		const Matrix Analysis::CalculateU( const Matrix& E, const Matrix& Q ) const {
#ifdef PROFILE_ANALYSIS
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
            Matrix retVal( E.Width, Q.Height );
			MatrixMultiply( E, false, Q, false, retVal );
            
#ifdef PROFILE_ANALYSIS
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[Analysis] Calculate U: " << elapsed << "ms" << std::endl;
#endif
			return retVal;
		}
		
		void Analysis::computeEigenvectorsFull( ContextImpl &contextImpl, const Parameters& params ) {
			timeval start, end;
			gettimeofday( &start, 0 );
			
			timeval tp_begin, tp_hess, tp_diag, tp_e, tp_s, tp_s_matrix, tp_q, tp_u;
			
			gettimeofday( &tp_begin, NULL );
			Context &context = contextImpl.getOwner();
			State state = context.getState( State::Positions | State::Forces );
            std::vector<Vec3> positions = state.getPositions();
			
			/*********************************************************************/
			/*                                                                   */
			/* Block Hessian Code (Cickovski/Sweet)                              */
			/*                                                                   */
			/*********************************************************************/
			
			// Initial residue data (where in OpenMM?)
			
			// For now, since OpenMM input files do not contain residue information
			// I am assuming that they will always start with the N-terminus, just for testing.
			// This is true for the villin.xml but may not be true in the future.
			// need it to parallelize.
			
			if( !mInitialized ) {
				Initialize( context, params );
			}
			
			int n = 3 * mParticleCount;
			
			// Copy the positions.
            std::vector<Vec3> blockPositions;
			for( unsigned int i = 0; i < mParticleCount; i++ ) {
				Vec3 atom( state.getPositions()[i][0], state.getPositions()[i][1], state.getPositions()[i][2] );
				blockPositions.push_back( atom );
			}
			
			blockContext->setPositions( blockPositions );
			/*********************************************************************/
			
#ifdef FIRST_ORDER
			vector<Vec3> block_start_forces = blockContext->getState( State::Forces ).getForces();
#endif
			
			Matrix h( n, n );
			std::vector<Vec3> initialBlockPositions( blockPositions );
			for( unsigned int i = 0; i < mLargestBlockSize; i++ ) {
				// Perturb the ith degree of freedom in EACH block
				// Note: not all blocks will have i degrees, we have to check for this
				for( unsigned int j = 0; j < blocks.size(); j++ ) {
					unsigned int dof_to_perturb = 3 * blocks[j] + i;
					unsigned int atom_to_perturb = dof_to_perturb / 3;  // integer trunc
					
					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= mParticleCount ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}
					
					blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialBlockPositions[atom_to_perturb][dof_to_perturb % 3] - params.blockDelta;
				}
				
				blockContext->setPositions( blockPositions );
				std::vector<Vec3> forces1 = blockContext->getState( State::Forces ).getForces();
				
#ifndef FIRST_ORDER
				// Now, do it again...
				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc
					
					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= mParticleCount ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}
					
					blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialBlockPositions[atom_to_perturb][dof_to_perturb % 3] + params.blockDelta;
				}
				
				blockContext->setPositions( blockPositions );
				std::vector<Vec3> forces2 = blockContext->getState( State::Forces ).getForces();
#endif
				
				// revert block positions
				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc
					
					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= mParticleCount ) {
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
					if( j == blocks.size() - 1 && atom_to_perturb >= mParticleCount ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}
					
					int col = dof_to_perturb;
					
					int start_dof = 3 * blocks[j];
					int end_dof;
					if( j == blocks.size() - 1 ) {
						end_dof = 3 * mParticleCount;
					} else {
						end_dof = 3 * blocks[j + 1];
					}
					
					for( int k = start_dof; k < end_dof; k++ ) {
#ifdef FIRST_ORDER
						double blockscale = 1.0 / ( params.blockDelta * sqrt( mParticleMass[atom_to_perturb] * mParticleMass[k / 3] ) );
						h(k,col) = ( forces1[k / 3][k % 3] - block_start_forces[k / 3][k % 3] ) * blockscale;
#else
						double blockscale = 1.0 / ( 2 * params.blockDelta * sqrt( mParticleMass[atom_to_perturb] * mParticleMass[k / 3] ) );
						h(k,col) = ( forces1[k / 3][k % 3] - forces2[k / 3][k % 3] ) * blockscale;
#endif
					}
				}
			}
			
			gettimeofday( &tp_hess, NULL );
			
			const double hessElapsed = ( tp_hess.tv_sec - tp_begin.tv_sec ) * 1000.0 + ( tp_hess.tv_usec - tp_begin.tv_usec ) / 1000.0;
			std::cout << "Time to compute hessian: " << hessElapsed << "ms" << std::endl;
			
			// Make sure it is exactly symmetric.
			for( int i = 0; i < n; i++ ) {
				for( int j = 0; j < i; j++ ) {
					double avg = 0.5f * ( h(i,j) + h(j,i) );
					h(i,j) = avg;
					h(j,i) = avg;
				}
			}
			
			//WriteHessian( h );
			
			// Diagonalize each block Hessian, get Eigenvectors
			// Note: The eigenvalues will be placed in one large array, because
			//       we must sort them to get k
			
            std::vector<double> block_eigval( n );
			Matrix block_eigvec( n, n );
			
			DiagonalizeBlocks( h, positions, block_eigval, block_eigvec );
			
			gettimeofday( &tp_diag, NULL );
			
			const double diagElapsed = ( tp_diag.tv_sec - tp_hess.tv_sec ) * 1000.0 + ( tp_diag.tv_usec - tp_hess.tv_usec ) / 1000.0;
            std::cout << "Time to diagonalize block hessian: " << diagElapsed << "ms" << std::endl;
			
			//***********************************************************
			// This section here is only to find the cuttoff eigenvalue.
			// First sort the eigenvectors by the absolute value of the eigenvalue.
			
			// sort all eigenvalues by absolute magnitude to determine cutoff
			std::vector<EigenvalueColumn> sortedEvalues = SortEigenvalues( block_eigval );
			
			int max_eigs = params.bdof * blocks.size();
			double cutEigen = sortedEvalues[max_eigs].first;  // This is the cutoff eigenvalue
			
			// get cols of all eigenvalues under cutoff
			std::vector<int> selectedEigsCols;
			for( int i = 0; i < n; i++ ) {
				if( fabs( block_eigval[i] ) < cutEigen ) {
					selectedEigsCols.push_back( i );
				}
			}
			
			// we may select fewer eigs if there are duplicate eigenvalues
			const int m = selectedEigsCols.size();
			
			// Inefficient, needs to improve.
			// Basically, just setting up E and E^T by
			// copying values from bigE.
			// Again, right now I'm only worried about
			// correctness plus this time will be marginal compared to
			// diagonalization.
			Matrix E( n, m );
			for( int i = 0; i < m; i++ ) {
				int eig_col = selectedEigsCols[i];
				for( int j = 0; j < n; j++ ) {
					E(j,i) = block_eigvec(j,eig_col);
				}
			}
			
			gettimeofday( &tp_e, NULL );
			
			const double eElapsed = ( tp_e.tv_sec - tp_diag.tv_sec ) * 1000.0 + ( tp_e.tv_usec - tp_diag.tv_usec ) / 1000.0;
			std::cout << "Time to compute E: " << eElapsed << "ms" << std::endl;
			
			//WriteBlockEigs( E );
			
			//*****************************************************************
			// Compute S, which is equal to E^T * H * E.
			Matrix S( m, m );
			Matrix HE( n, m );
			// Compute eps.
			const double eps = params.sDelta;
			
			// Make a temp copy of positions.
			std::vector<Vec3> tmppos( positions );
			
#ifdef FIRST_ORDER
			vector<Vec3> forces_start = context.getState( State::Forces ).getForces();
#endif
			
			// Loop over i.
			for( unsigned int k = 0; k < m; k++ ) {
				// Perturb positions.
				int pos = 0;
				
				// forward perturbations
				for( unsigned int i = 0; i < mParticleCount; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j] + eps * E(3 * i + j,k) / sqrt( mParticleMass[i] );
						pos++;
					}
				}
				context.setPositions( tmppos );
				
				// Calculate F(xi).
				std::vector<Vec3> forces_forward = context.getState( State::Forces ).getForces();
				
#ifndef FIRST_ORDER
				// backward perturbations
				for( unsigned int i = 0; i < mParticleCount; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j] - eps * E(3 * i + j,k) / sqrt( mParticleMass[i] );
					}
				}
				context.setPositions( tmppos );
				
				// Calculate forces
				std::vector<Vec3> forces_backward = context.getState( State::Forces ).getForces();
#endif
				
				for( int i = 0; i < n; i++ ) {
#ifdef FIRST_ORDER
					const double scaleFactor = sqrt( mParticleMass[i / 3] ) * 1.0 * eps;
					HE(i,k) = ( forces_forward[i / 3][i % 3] - forces_start[i / 3][i % 3] ) / scaleFactor;
#else
					const double scaleFactor = sqrt( mParticleMass[i / 3] ) * 2.0 * eps;
					HE(i,k) = ( forces_forward[i / 3][i % 3] - forces_backward[i / 3][i % 3] ) / scaleFactor;
#endif
				}
				
				// restore positions
				for( unsigned int i = 0; i < mParticleCount; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j];
					}
				}
			}
			
			// *****************************************************************
			// restore unperturbed positions
			context.setPositions( positions );
			
			gettimeofday( &tp_s, NULL );
			
			const double sElapsed = ( tp_s.tv_sec - tp_e.tv_sec ) * 1000.0 + ( tp_s.tv_usec - tp_e.tv_usec ) / 1000.0;
			std::cout << "Time to compute S: " << sElapsed << "ms" << std::endl;
			
			MatrixMultiply( E, true, HE, false, S );
			
			//WriteS(S);
			
			// make S symmetric
			for( unsigned int i = 0; i < S.Width; i++ ) {
				for( unsigned int j = 0; j < S.Height; j++ ) {
					double avg = 0.5f * ( S(i,j) + S(j,i) );
					S(i,j) = avg;
					S(j,i) = avg;
				}
			}
			
			gettimeofday( &tp_s_matrix, NULL );
			
			const double sMatrixElapsed = ( tp_s_matrix.tv_sec - tp_s.tv_sec ) * 1000.0 + ( tp_s_matrix.tv_usec - tp_s.tv_usec ) / 1000.0;
			std::cout << "Time to compute matrix S: " << sMatrixElapsed << "ms" << std::endl;
			
			// Diagonalizing S by finding eigenvalues and eigenvectors...
            std::vector<double> dS( m );
			Matrix q( m, m );
			FindEigenvalues( S, dS, q );
			
			// Sort by ABSOLUTE VALUE of eigenvalues.
			sortedEvalues = SortEigenvalues( dS );
			
			Matrix Q( q.Height, q.Width );
			for( int i = 0; i < sortedEvalues.size(); i++ ){
				for( int j = 0; j < q.Height; j++ ) {
					Q(j,i) = q(j,sortedEvalues[i].second);
				}
			}
			
			gettimeofday( &tp_q, NULL );
			
			const double qElapsed = ( tp_q.tv_sec - tp_s.tv_sec ) * 1000.0 + ( tp_q.tv_usec - tp_s.tv_usec ) / 1000.0;
			std::cout << "Time to compute Q: " << qElapsed << "ms" << std::endl;
			
			Matrix U = CalculateU( E, Q );
			
			gettimeofday( &tp_u, NULL );
			
			const double uElapsed = ( tp_u.tv_sec - tp_q.tv_sec ) * 1000.0 + ( tp_u.tv_usec - tp_q.tv_usec ) / 1000.0;
			std::cout << "Time to compute U: " << uElapsed << "ms" << std::endl;
			
			const unsigned int modes = params.modes;
			
			//WriteModes( U, modes );
			
			eigenvectors.resize( modes, std::vector<Vec3>( mParticleCount ) );
			for( unsigned int i = 0; i < modes; i++ ) {
				for( unsigned int j = 0; j < mParticleCount; j++ ) {
					eigenvectors[i][j] = Vec3( U(3 * j, i), U(3 * j + 1, i), U(3 * j + 2,i) );
				}
			}
			
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[Analysis] Compute Eigenvectors: " << elapsed << "ms" << std::endl;
		}
				
		void Analysis::Initialize( Context &context, const Parameters &params ) {
#ifdef PROFILE_ANALYSIS
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			
			// Get Current System
			System &system = context.getSystem();
			
			// Store Particle Information
			mParticleCount = context.getState( State::Positions ).getPositions().size();
			
			mParticleMass.reserve( mParticleCount );
			for( unsigned int i = 0; i < mParticleCount; i++ ){
				mParticleMass.push_back( system.getParticleMass( i ) );
			}
			
			// Create New System
			System *blockSystem = new System();
			std::cout << "res per block " << params.res_per_block << std::endl;
			for( int i = 0; i < mParticleCount; i++ ) {
				blockSystem->addParticle( mParticleMass[i] );
			}
			
			int block_start = 0;
			for( int i = 0; i < params.residue_sizes.size(); i++ ) {
				if( i % params.res_per_block == 0 ) {
					blocks.push_back( block_start );
				}
				block_start += params.residue_sizes[i];
			}
			
			for( int i = 1; i < blocks.size(); i++ ) {
				int block_size = blocks[i] - blocks[i - 1];
				if( block_size > mLargestBlockSize ) {
					mLargestBlockSize = block_size;
				}
			}
			
			mLargestBlockSize *= 3; // degrees of freedom in the largest block
			std::cout << "blocks " << blocks.size() << std::endl;
			std::cout << blocks[blocks.size() - 1] << std::endl;
			
			// Creating a whole new system called the blockSystem.
			// This system will only contain bonds, angles, dihedrals, and impropers
			// between atoms in the same block.
			// Also contains pairwise force terms which are zeroed out for atoms
			// in different blocks.
			// This necessitates some copying from the original system, but is required
			// because OpenMM populates all data when it reads XML.
			// Copy all atoms into the block system.
			
			// Copy the center of mass force.
			std::cout << "adding forces..." << std::endl;
			for( int i = 0; i < params.forces.size(); i++ ) {
				std::string forcename = params.forces[i].name;
				std::cout << "Adding force " << forcename << " at index " << params.forces[i].index << std::endl;
				if( forcename == "CenterOfMass" ) {
					blockSystem->addForce( &system.getForce( params.forces[i].index ) );
				} else if( forcename == "Bond" ) {
					// Create a new harmonic bond force.
					// This only contains pairs of atoms which are in the same block.
					// I have to iterate through each bond from the old force, then
					// selectively add them to the new force based on this condition.
					HarmonicBondForce *hf = new HarmonicBondForce();
					const HarmonicBondForce *ohf = dynamic_cast<const HarmonicBondForce *>( &system.getForce( params.forces[i].index ) );
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
					const HarmonicAngleForce *ahf = dynamic_cast<const HarmonicAngleForce *>( &system.getForce( params.forces[i].index ) );
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
					const PeriodicTorsionForce *optf = dynamic_cast<const PeriodicTorsionForce *>( &system.getForce( params.forces[i].index ) );
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
					const RBTorsionForce *orbtf = dynamic_cast<const RBTorsionForce *>( &system.getForce( params.forces[i].index ) );
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
					const NonbondedForce *nbf = dynamic_cast<const NonbondedForce *>( &system.getForce( params.forces[i].index ) );
					
					cbf->addPerBondParameter( "q" );
					cbf->addPerBondParameter( "sigma" );
					cbf->addPerBondParameter( "eps" );
					
					// store exceptions
					// exceptions[p1][p2] = params
					std::map<int, std::map<int, std::vector<double> > > exceptions;
					
					for( int i = 0; i < nbf->getNumExceptions(); i++ ) {
						int p1, p2;
						double q, sig, eps;
						nbf->getExceptionParameters( i, p1, p2, q, sig, eps );
						if( inSameBlock( p1, p2 ) ) {
							std::vector<double> params;
							params.push_back( q );
							params.push_back( sig );
							params.push_back( eps );
							if( exceptions.count( p1 ) == 0 ) {
								std::map<int, std::vector<double> > pair_exception;
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
								std::vector<double> params = exceptions[i][j];
								cbf->addBond( i, j, params );
							}
							// no exception, normal interaction
							else {
								std::vector<double> params;
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
					std::cout << "Unknown Force: " << forcename << std::endl;
				}
			}
			std::cout << "done." << std::endl;
			
			VerletIntegrator *integ = new VerletIntegrator( 0.000001 );
			if( blockContext ) {
				delete blockContext;
			}
            
            std::string sPlatform = "";
            switch( params.BlockDiagonalizePlatform ){
				case Preference::Reference:
					sPlatform = "Reference";
					break;
				case Preference::OpenCL:
					sPlatform = "OpenCL";
					break;
                case Preference::CUDA:
                    sPlatform = "Cuda";
					break;
			}
            
            OpenMM::Platform& platform = OpenMM::Platform::getPlatformByName( sPlatform );
            
            if( params.BlockDiagonalizePlatform != 0 && params.DeviceID != -1 ){
                std::ostringstream stream;
                stream << params.DeviceID;
                
                std::cout << "OpenMM Block Device: " << params.DeviceID << std::endl;
                
                if( params.BlockDiagonalizePlatform == 1 ){
                    platform.setPropertyDefaultValue("OpenCLDeviceIndex", stream.str() );
                }else{
                    platform.setPropertyDefaultValue("CudaDevice", stream.str() );
                }
            }
            
            blockContext = new Context( *blockSystem, *integ, platform );
			
			mInitialized = true;
			
#ifdef PROFILE_ANALYSIS
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[Analysis] Initialize: " << elapsed << "ms" << std::endl;
#endif
		}
		
		void Analysis::DiagonalizeBlocks( const Matrix& hessian, const std::vector<Vec3>& positions, std::vector<double>& eval, Matrix& evec ){
			std::vector<Block> HTilde( blocks.size() );
			
			// Create Blocks
			for( int i = 0; i < blocks.size(); i++ ) {
				// Find block size
				const int startatom = 3 * blocks[i];
				int endatom = 3 * mParticleCount - 1 ;
				
				if( i != ( blocks.size() - 1 ) ) {
					endatom = 3 * blocks[i + 1] - 1;
				}
				
				HTilde[i].StartAtom = startatom;
				HTilde[i].EndAtom = endatom;
				HTilde[i].Data = Matrix( endatom - startatom + 1, endatom - startatom + 1 );

				// Copy data from big hessian
				for( int j = startatom; j <= endatom; j++ ) {
					for( int k = startatom; k <= endatom; k++ ) {
						HTilde[i].Data(k - startatom,j - startatom) = hessian(k,j);
					}
				}
			}
			
			// Diagonalize Blocks
			#pragma omp parallel for
			for( int i = 0; i < blocks.size(); i++ ) {
				printf( "Diagonalizing Block: %d\n", i );
				DiagonalizeBlock( HTilde[i], positions, mParticleMass, eval, evec );
				GeometricDOF( HTilde[i].Data.Width, HTilde[i].StartAtom, HTilde[i].EndAtom, positions, mParticleMass, eval, evec );
			}
		}
		
		void Analysis::DiagonalizeBlock( const Block& block, const std::vector<Vec3>& positions, const std::vector<double>& Mass, std::vector<double>& eval, Matrix& evec ) {
			const unsigned int size = block.Data.Width;
			
			// 3. Diagonalize the block Hessian only, and get eigenvectors
            std::vector<double> di( size );
			Matrix Qi( size, size );
			FindEigenvalues( block.Data, di, Qi );
			
			for( int j = 0; j < size; j++ ) {
				eval[block.StartAtom + j] = di[j];
				for( int k = 0; k < size; k++ ) {
					evec(block.StartAtom + k,block.StartAtom + j) = Qi(k,j);
				}
			}
		}
		
		void Analysis::GeometricDOF( const int size, const int start, const int end, const std::vector<Vec3>& positions, const std::vector<double>& Mass, std::vector<double>& eval, Matrix& evec ) {
			// find geometric dof
            std::vector<double> values( size );
			for( int i = 0; i < size; i++ ){
				values[i] = eval[start + i];
			}

			Matrix Qi_gdof( size, size );
			
			Vec3 pos_center( 0.0, 0.0, 0.0 );
			double totalmass = 0.0;
			
			int test = 0;
			for( int j = start; j < end; j += 3 ) {
				double mass = Mass[ j / 3 ];
				pos_center += positions[j / 3] * mass;
				totalmass += mass;
				test++;
			}
			
			double norm = sqrt( totalmass );
			
			// actual center
			pos_center *= 1.0 / totalmass;
			
			// create geometric dof vectors
			// iterating over rows and filling in values for 6 vectors as we go
			for( int j = 0; j < size; j += 3 ) {
				double atom_index = ( start + j ) / 3;
				double mass = Mass[atom_index];
				double factor = sqrt( mass ) / norm;
				
				// translational
				Qi_gdof(j,0)   = factor;
				Qi_gdof(j + 1,1) = factor;
				Qi_gdof(j + 2,2) = factor;
				
				// rotational
				// cross product of rotation axis and vector to center of molecule
				// x-axis (b1=1) ja3-ka2
				// y-axis (b2=1) ka1-ia3
				// z-axis (b3=1) ia2-ja1
				Vec3 diff = positions[atom_index] - pos_center;
				// x
				Qi_gdof(j + 1,3) =  diff[2] * factor;
				Qi_gdof(j + 2,3) = -diff[1] * factor;
				
				// y
				Qi_gdof(j,4)   = -diff[2] * factor;
				Qi_gdof(j + 2,4) =  diff[0] * factor;
				
				// z
				Qi_gdof(j,5)   =  diff[1] * factor;
				Qi_gdof(j + 1,5) = -diff[0] * factor;
			}
			
			// normalize first rotational vector
			double rotnorm = 0.0;
			for( int j = 0; j < size; j++ ) {
				rotnorm += Qi_gdof(j,3) * Qi_gdof(j,3);
			}
			
			rotnorm = 1.0 / sqrt( rotnorm );
			
			for( int j = 0; j < size; j++ ) {
				Qi_gdof(j,3) = Qi_gdof(j,3) * rotnorm;
			}
			
			// orthogonalize rotational vectors 2 and 3
			for( int j = 4; j < ConservedDegreesOfFreedom; j++ ) { // <-- vector we're orthogonalizing
				for( int k = 3; k < j; k++ ) { // <-- vectors we're orthognalizing against
					double dot_prod = 0.0;
					for( int l = 0; l < size; l++ ) {
						dot_prod += Qi_gdof(l,k) * Qi_gdof(l,j);
					}
					for( int l = 0; l < size; l++ ) {
						Qi_gdof(l,j) = Qi_gdof(l,j) - Qi_gdof(l,k) * dot_prod;
					}
				}
				
				// normalize residual vector
				double rotnorm = 0.0;
				for( int l = 0; l < size; l++ ) {
					rotnorm += Qi_gdof(l,j) * Qi_gdof(l,j);
				}
				
				rotnorm = 1.0 / sqrt( rotnorm );
				
				for( int l = 0; l < size; l++ ) {
					Qi_gdof(l,j) = Qi_gdof(l,j) * rotnorm;
				}
			}
			
			// sort eigenvalues by absolute magnitude
			std::vector<EigenvalueColumn> sortedPairs = SortEigenvalues( values );
			
			// orthogonalize original eigenvectors against gdof
			// number of evec that survive orthogonalization
			unsigned int curr_evec = ConservedDegreesOfFreedom;
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
				int col = sortedPairs.at( j ).second;
				
				// copy original vector to Qi_gdof -- updated in place
				for( int l = 0; l < size; l++ ) {
					Qi_gdof(l,curr_evec) = evec(start+l,start+col);
				}
				
				// get dot products with previous vectors
				for( int k = 0; k < curr_evec; k++ ) { // <-- vector orthog against
					// dot product between original vector and previously
					// orthogonalized vectors
					double dot_prod = 0.0;
					for( int l = 0; l < size; l++ ) {
						dot_prod += Qi_gdof(l,k) * evec(start+l,start+col);
					}
					
					// subtract from current vector -- update in place
					for( int l = 0; l < size; l++ ) {
						Qi_gdof(l,curr_evec) = Qi_gdof(l,curr_evec) - Qi_gdof(l,k) * dot_prod;
					}
				}
				
				//normalize residual vector
				double norm = 0.0;
				for( int l = 0; l < size; l++ ) {
					norm += Qi_gdof(l,curr_evec) * Qi_gdof(l,curr_evec);
				}
				
				// if norm less than 1/20th of original
				// continue on to next vector
				// we don't update curr_evec so this vector
				// will be overwritten
				if( norm < 0.05 ) {
					continue;
				}
				
				// scale vector
				norm = sqrt( norm );
				for( int l = 0; l < size; l++ ) {
					Qi_gdof(l,curr_evec) = Qi_gdof(l,curr_evec) / norm;
				}
				
				curr_evec++;
			}
			
			// 4. Copy eigenpairs to big array
			//    This is necessary because we have to sort them, and determine
			//    the cutoff eigenvalue for everybody.
			// we assume curr_evec <= size
			for( int j = 0; j < curr_evec; j++ ) {
				int col = sortedPairs.at( j ).second;
				eval[start + j] = values[col];
				
				// orthogonalized eigenvectors already sorted by eigenvalue
				for( int k = 0; k < size; k++ ) {
					evec(start + k,start + j) = Qi_gdof(k,j);
				}
			}
		}
	}
}
