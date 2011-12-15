#include "LTMD/Math.h"

#ifdef INTEL_MKL
	#include <mkl_blas.h>
	#include <mkl_lapack.h>
#else
	#include "tnt_array2d_utils.h"
#endif

#include <vector>

void MatrixMultiply( const TNT::Array2D<double>& matrixA, const TNT::Array2D<double>& matrixB, TNT::Array2D<double>& matrixC ) {
	#ifdef INTEL_MKL
		int m = matrixA.dim1();                                                                                                                                                                                                                                                   
		int k = matrixA.dim2();                                                                                                                                                                                                                                                   
		int n = matrixB.dim2();                                                                                                                                                                                                                                                   
		char transa = 'T';                                                                                                                                                                                                                                                        
		char transb = 'T';                                                                                                                                                                                                                                                        
		double alpha = 1.0;                                                                                                                                                                                                                                                       
		double beta = 0.0;                                                                                                                                                                                                                                                        
		int lda = m;                                                                                                                                                                                                                                                              
		int ldb = k;                                                                                                                                                                                                                                                              
		int ldc = m;                                                                                                                                                                                                                                                              
		std::vector<double> a(m*k);                                                                                                                                                                                                                                                    
		std::vector<double> b(k*n);                                                                                                                                                                                                                                                    
		std::vector<double> c(m*n);
																																												
		for (int i = 0; i < m; i++){                                                                                                                                                                                                                                        
			for (int j = 0; j < k; j++) {
				a[i*k+j] = matrixA[i][j];
			}
		}
																																																																																																																															
		for (int i = 0; i < k; i++){                                                                                                                                                                                                                                              
			for (int j = 0; j < n; j++){                                                                                                                                                                                                                                         
			   b[i*n+j] = matrixB[i][j];
			}
		}
																																																																			 
		dgemm(&transa, &transb, &m, &n, &k, &alpha, &a[0], &lda, &b[0], &ldb, &beta, &c[0], &ldc);                                                                                                                                                                            
																																																																			 
		for (int i = 0; i < m; i++){                                                                                                                                                                                                                                               
			for (int j = 0; j < n; j++){                                                                                                                                                                                                                                            
			  matrixC[j][i] = c[i*n+j];
			}
		}
	#else
		matrixC = matmult( matrixA, matrixB );
	#endif
}

void FindEigenvalues( const TNT::Array2D<double>& matrix, TNT::Array1D<double>& values, TNT::Array2D<double>& vectors ) {
	#ifdef INTEL_MKL
		int n = matrix.dim1();
		char jobz = 'V';
		char uplo = 'U';
		int lwork = 3*n-1;
		std::vector<double> a(n*n);
		std::vector<double> w(n);
		std::vector<double> work(lwork);
		int info;
		
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++){
				a[i*n+j] = matrix[i][j];
			}
		}
				
		dsyev(&jobz, &uplo, &n, &a[0], &n, &w[0], &work[0], &lwork, &info);
		
		for (int i = 0; i < n; i++) values[i] = w[i];
		
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++){
				vectors[i][j] = -a[i+j*n];
			}
		}
	#else
		JAMA::Eigenvalue<double> decomp( matrix );
		decomp.getRealEigenvalues( values );
		decomp.getV( vectors );
	#endif
}
