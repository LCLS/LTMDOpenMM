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
        
		// Function Declarations
        void WriteS(const TNT::Array2D<double>& S);
		void WriteHessian( const TNT::Array2D<double>& H );
		void WriteBlockEigs( const TNT::Array2D<double>& H );
		void WriteModes( const TNT::Array2D<double>& U, const unsigned int modes );
        
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
			retVal.reserve( values.dim() );
			for( unsigned int i = 0; i < values.dim(); i++ ){
				retVal.push_back( std::make_pair( std::fabs( values[i] ), i ) );
			}
			
			// Sort Data
			std::sort( retVal.begin(), retVal.end(), sort_func );
			
			return retVal;
		}
		
		const TNT::Array2D<double> Analysis::CalculateU( const TNT::Array2D<double>& E, const TNT::Array2D<double>& Q ) const {
#ifdef PROFILE_ANALYSIS
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			TNT::Array2D<double> retVal( E.dim1(), Q.dim2(), 0.0 );
            
			MatrixMultiply( E, Q, retVal );
			
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
			vector<Vec3> positions = state.getPositions();
            
			int n = 3 * mParticleCount;
            
            TNT::Array2D<double> h( n, n, 0.0 );
            std::vector<Vec3> blockPositions( positions );
            
            // Set position and calculate forces
            for( unsigned int i = 0; i < n; i++ ){
                // Perturb two backward
                blockPositions[i/3][i%3] = positions[i/3][i%3] - 2.0 * params.blockDelta;
                
                context.setPositions( blockPositions );
                vector<Vec3> forces1 = context.getState( State::Forces ).getForces();
                
                // Perturb backward
                blockPositions[i/3][i%3] = positions[i/3][i%3] - params.blockDelta;

                context.setPositions( blockPositions );
                vector<Vec3> forces2 = context.getState( State::Forces ).getForces();
                
                // Perturb forward
                blockPositions[i/3][i%3] = positions[i/3][i%3] + params.blockDelta;
                
                context.setPositions( blockPositions );
                vector<Vec3> forces3 = context.getState( State::Forces ).getForces();
                
                // Perturb two forward
                blockPositions[i/3][i%3] = positions[i/3][i%3] + 2.0 * params.blockDelta;
                
                context.setPositions( blockPositions );
                vector<Vec3> forces4 = context.getState( State::Forces ).getForces();

                // Revert
                blockPositions[i/3][i%3] = positions[i/3][i%3];

                // Fill matrix
                for( int j = 0; j < n; j++ ){
                    double blockscale = 1.0 / ( 12.0 * params.blockDelta * sqrt( mParticleMass[i/3] * mParticleMass[j / 3] ) );
                    h[j][i] = ( -forces1[j / 3][j % 3] + 8.0 * forces2[j / 3][j % 3] - 8.0 * forces3[j / 3][j % 3] + forces4[j / 3][j % 3]) * blockscale;
                }
            }

            context.setPositions(positions);
             
			gettimeofday( &tp_hess, NULL );
            
            const double hessElapsed = ( tp_hess.tv_sec - tp_begin.tv_sec ) * 1000.0 + ( tp_hess.tv_usec - tp_begin.tv_usec ) / 1000.0;
			cout << "Time to compute hessian: " << hessElapsed << "ms" << endl;
            
			// Make sure it is exactly symmetric.
			for( int i = 0; i < n; i++ ) {
				for( int j = 0; j < i; j++ ) {
					double avg = 0.5f * ( h[i][j] + h[j][i] );
					h[i][j] = avg;
                    h[j][i] = avg;
				}
			}
            
			WriteHessian( h );
            
			// Diagonalize each block Hessian, get Eigenvectors
			// Note: The eigenvalues will be placed in one large array, because
			//       we must sort them to get k
            
			TNT::Array1D<double> block_eigval( n, 0.0 );
			TNT::Array2D<double> block_eigvec( n, n, 0.0 );
            
            DiagonalizeBlocks( h, positions, block_eigval, block_eigvec );
            
			gettimeofday( &tp_diag, NULL );
            
            const double diagElapsed = ( tp_diag.tv_sec - tp_hess.tv_sec ) * 1000.0 + ( tp_diag.tv_usec - tp_hess.tv_usec ) / 1000.0;
			cout << "Time to diagonalize block hessian: " << diagElapsed << "ms" << endl;
            
			//***********************************************************
			// This section here is only to find the cuttoff eigenvalue.
			// First sort the eigenvectors by the absolute value of the eigenvalue.
            
			// sort all eigenvalues by absolute magnitude to determine cutoff
			std::vector<EigenvalueColumn> sortedEvalues = SortEigenvalues( block_eigval );
            
			int max_eigs = params.bdof * blocks.size();
			double cutEigen = sortedEvalues[max_eigs].first;  // This is the cutoff eigenvalue
			
			// get cols of all eigenvalues under cutoff
			vector<int> selectedEigsCols;
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
			TNT::Array2D<double> E( n, m, 0.0 );
			TNT::Array2D<double> E_transpose( m, n, 0.0 );
			for( int i = 0; i < m; i++ ) {
				int eig_col = selectedEigsCols[i];
				for( int j = 0; j < n; j++ ) {
					E_transpose[i][j] = block_eigvec[j][eig_col];
					E[j][i] = block_eigvec[j][eig_col];
				}
			}
            
			gettimeofday( &tp_e, NULL );
            
            const double eElapsed = ( tp_e.tv_sec - tp_diag.tv_sec ) * 1000.0 + ( tp_e.tv_usec - tp_diag.tv_usec ) / 1000.0;
            std::cout << "Time to compute E: " << eElapsed << "ms" << std::endl;
            
			WriteBlockEigs( E );
            
			//*****************************************************************
			// Compute S, which is equal to E^T * H * E.
			TNT::Array2D<double> S( m, m, 0.0 );
			TNT::Array2D<double> HE(n , m, 0.0);
			// Compute eps.
			const double eps = params.sDelta;
            
			// Make a temp copy of positions.
			vector<Vec3> tmppos( positions );
            
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
						tmppos[i][j] = positions[i][j] + eps * E[3 * i + j][k] / sqrt( mParticleMass[i] );
						pos++;
					}
				}
				context.setPositions( tmppos );
                
				// Calculate F(xi).
				vector<Vec3> forces_forward = context.getState( State::Forces ).getForces();
                
#ifndef FIRST_ORDER
				// backward perturbations
				for( unsigned int i = 0; i < mParticleCount; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j] - eps * E[3 * i + j][k] / sqrt( mParticleMass[i] );
					}
				}
				context.setPositions( tmppos );
                
				// Calculate forces
				vector<Vec3> forces_backward = context.getState( State::Forces ).getForces();
#endif
                
				for( int i = 0; i < n; i++ ) {
#ifdef FIRST_ORDER
                    const double scaleFactor = sqrt( mParticleMass[i / 3] ) * 1.0 * eps;
                    HE[i][k] = ( forces_forward[i / 3][i % 3] - forces_start[i / 3][i % 3] ) / scaleFactor;
#else
					const double scaleFactor = sqrt( mParticleMass[i / 3] ) * 2.0 * eps;
					HE[i][k] = ( forces_forward[i / 3][i % 3] - forces_backward[i / 3][i % 3] ) / scaleFactor;
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
			cout << "Time to compute S: " << sElapsed << "ms" << endl;
            
			MatrixMultiply( E_transpose, HE, S );
            
			WriteS(S);
            
			// make S symmetric
			for( unsigned int i = 0; i < S.dim1(); i++ ) {
				for( unsigned int j = 0; j < S.dim2(); j++ ) {
					double avg = 0.5f * ( S[i][j] + S[j][i] );
					S[i][j] = avg;
					S[j][i] = avg;
				}
			}
            
			gettimeofday( &tp_s_matrix, NULL );
            
            const double sMatrixElapsed = ( tp_s_matrix.tv_sec - tp_s.tv_sec ) * 1000.0 + ( tp_s_matrix.tv_usec - tp_s.tv_usec ) / 1000.0;
			cout << "Time to compute matrix S: " << sMatrixElapsed << "ms" << endl;
            
			// Diagonalizing S by finding eigenvalues and eigenvectors...
			TNT::Array1D<double> dS( m, 0.0 );
			TNT::Array2D<double> q( m, m, 0.0 );
			FindEigenvalues( S, dS, q );
            
			// Sort by ABSOLUTE VALUE of eigenvalues.
			sortedEvalues = SortEigenvalues( dS );
			
			TNT::Array2D<double> Q( q.dim2(), q.dim1(), 0.0 );
			for( int i = 0; i < sortedEvalues.size(); i++ ){
				for( int j = 0; j < q.dim2(); j++ ) {
					Q[j][i] = q[j][sortedEvalues[i].second];
				}
			}
            
			gettimeofday( &tp_q, NULL );
            
            const double qElapsed = ( tp_q.tv_sec - tp_s.tv_sec ) * 1000.0 + ( tp_q.tv_usec - tp_s.tv_usec ) / 1000.0;
			cout << "Time to compute Q: " << qElapsed << "ms" << endl;
			
			TNT::Array2D<double> U = CalculateU( E, Q );
			
			gettimeofday( &tp_u, NULL );
            
            const double uElapsed = ( tp_u.tv_sec - tp_q.tv_sec ) * 1000.0 + ( tp_u.tv_usec - tp_q.tv_usec ) / 1000.0;
			cout << "Time to compute U: " << uElapsed << "ms" << endl;
            
			const unsigned int modes = params.modes;
            
			WriteModes( U, modes );
            
			eigenvectors.resize( modes, vector<Vec3>( mParticleCount ) );
			for( unsigned int i = 0; i < modes; i++ ) {
				for( unsigned int j = 0; j < mParticleCount; j++ ) {
					eigenvectors[i][j] = Vec3( U[3 * j][i], U[3 * j + 1][i], U[3 * j + 2][i] );
				}
			}
            
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[Analysis] Compute Eigenvectors: " << elapsed << "ms" << std::endl;
		}
        
        void WriteS(const TNT::Array2D<double>& S)
        {
#ifdef VALIDATION
            std::ofstream file("s.txt");
            file.precision(10);
            for( unsigned int i = 0; i < S.dim2(); i++ ) {
                for( unsigned int j = 0; j < S.dim1(); j++ ) {
                    file << j << " " << i << " " << S[j][i] << std::endl;
                }
            }
#endif
        }
		
		void WriteHessian( const TNT::Array2D<double>& H ) {
#ifdef VALIDATION
			std::ofstream file( "block_hessian.txt" );
			file.precision( 10 );
			for( unsigned int i = 0; i < H.dim2(); i++ ) {
				for( unsigned int j = 0; j < H.dim1(); j++ ) {
					file << j << " " << i << " " << H[j][i] << std::endl;
				}
			}
#endif
		}
		
		
		void WriteBlockEigs( const TNT::Array2D<double>& matrix ) {
#ifdef VALIDATION
			std::ofstream file( "block_eigs.txt" );
			file.precision( 10 );
			for( unsigned int i = 0; i < matrix.dim2(); i++ ) {
				for( unsigned int j = 0; j < matrix.dim1(); j++ ) {
					file << j << " " << i << " " << matrix[j][i] << std::endl;
				}
			}
#endif
		}
        
		void WriteModes( const TNT::Array2D<double>& U, const unsigned int modes ) {
#ifdef VALIDATION
			std::ofstream file( "eigenvectors.txt" );
			file.precision( 10 );
			for( unsigned int i = 0; i < modes; i++ ) {
				for( unsigned int j = 0; j < U.dim1(); j++ ) {
					file << j << " " << i << " " << U[j][i] << std::endl;
				}
			}
#endif
		}
        
        void Analysis::DiagonalizeBlocks( const TNT::Array2D<double>& hessian, const std::vector<Vec3>& positions, TNT::Array1D<double>& eval, TNT::Array2D<double>& evec ){
            std::vector<Block> HTilde( blocks.size() );
            
            // Create Blocks
            for( int i = 0; i < blocks.size(); i++ ) {
                // Find block size
                const int startatom = 3 * blocks[i];
                int endatom = 3 * mParticleCount - 1 ;
                
                if( i != ( blocks.size() - 1 ) ) {
                    endatom = 3 * blocks[i + 1] - 1;
                }
                
                const int size = endatom - startatom + 1;
                
                // Copy data from big hessian
                TNT::Array2D<double> h_tilde( size, size, 0.0 );
                for( int j = startatom; j <= endatom; j++ ) {
                    for( int k = startatom; k <= endatom; k++ ) {
                        h_tilde[k - startatom][j - startatom] = hessian[k][j];
                    }
                }
                
                HTilde[i].StartAtom = startatom;
                HTilde[i].EndAtom = endatom;
                HTilde[i].Data = h_tilde;
            }
            
            // Diagonalize Blocks
#pragma omp parallel for
            for( int i = 0; i < blocks.size(); i++ ) {
                printf( "Diagonalizing Block: %d\n", i );
                DiagonalizeBlock( HTilde[i], positions, mParticleMass, eval, evec );
                GeometricDOF( HTilde[i].Data.dim1(), HTilde[i].StartAtom, HTilde[i].EndAtom, positions, mParticleMass, eval, evec );
            }
        }
		
        void Analysis::DiagonalizeBlock( const Block& block, const std::vector<Vec3>& positions, const std::vector<double>& Mass, TNT::Array1D<double>& eval, TNT::Array2D<double>& evec ) {
            const unsigned int size = block.Data.dim1();
            
			// 3. Diagonalize the block Hessian only, and get eigenvectors
			TNT::Array1D<double> di( size, 0.0 );
			TNT::Array2D<double> Qi( size, size, 0.0 );
			FindEigenvalues( block.Data, di, Qi );
            
            for( int j = 0; j < size; j++ ) {
				eval[block.StartAtom + j] = di[j];
				for( int k = 0; k < size; k++ ) {
					evec[block.StartAtom + k][block.StartAtom + j] = Qi[k][j];
				}
			}
		}
        
        void Analysis::GeometricDOF( const int size, const int start, const int end, const std::vector<Vec3>& positions, const std::vector<double>& Mass, TNT::Array1D<double>& eval, TNT::Array2D<double>& evec ) {
            // find geometric dof
            TNT::Array1D<double> values = eval;
			TNT::Array2D<double> Qi_gdof( size, size, 0.0 );
            
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
			for( int j = 4; j < ConservedDegreesOfFreedom; j++ ) { // <-- vector we're orthogonalizing
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
				}
                
				rotnorm = 1.0 / sqrt( rotnorm );
                
				for( int l = 0; l < size; l++ ) {
					Qi_gdof[l][j] = Qi_gdof[l][j] * rotnorm;
				}
			}
            
            // sort eigenvalues by absolute magnitude
			std::vector<EigenvalueColumn> sortedPairs = SortEigenvalues( values );
            
			// orthogonalize original eigenvectors against gdof
			// number of evec that survive orthogonalization
			int curr_evec = ConservedDegreesOfFreedom;
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
					Qi_gdof[l][curr_evec] = evec[l][col];
				}
                
				// get dot products with previous vectors
				for( int k = 0; k < curr_evec; k++ ) { // <-- vector orthog against
					// dot product between original vector and previously
					// orthogonalized vectors
					double dot_prod = 0.0;
					for( int l = 0; l < size; l++ ) {
						dot_prod += Qi_gdof[l][k] * evec[l][col];
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
					continue;
				}
                
				// scale vector
				norm = sqrt( norm );
				for( int l = 0; l < size; l++ ) {
					Qi_gdof[l][curr_evec] = Qi_gdof[l][curr_evec] / norm;
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
					evec[start + k][start + j] = Qi_gdof[k][j];
				}
			}
        }
	}
}
