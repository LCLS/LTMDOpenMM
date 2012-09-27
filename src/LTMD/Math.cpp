#include "LTMD/Math.h"

#ifdef INTEL_MKL
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#endif

#include "tnt_array2d_utils.h"

#include <vector>

void MatrixMultiply( const TNT::Array2D<double>& matrixA, const TNT::Array2D<double>& matrixB, TNT::Array2D<double>& matrixC ) {
#ifdef INTEL_MKL
  std::vector<double> a( matrixA.dim1() * matrixA.dim2() );

  for( int i = 0; i < matrixA.dim1(); i++ ) {
    for( int j = 0; j < matrixA.dim2(); j++ ) {
      a[i * matrixA.dim2() + j] = matrixA[i][j];
    }
  }

  std::vector<double> b( matrixB.dim1() * matrixB.dim2());

	for( int i = 0; i < matrixB.dim1(); i++ ) {
    for( int j = 0; j < matrixB.dim2(); j++ ) {
      b[i * matrixB.dim2() + j] = matrixB[i][j];
    }
  }

	const int m = matrixA.dim1();
	const int k = matrixA.dim2();
	const int n = matrixB.dim2();

	const char transa = 'T';
	const char transb = 'T';
	const double alpha = 1.0;
	const double beta = 0.0;
	const int lda = k;
	const int ldb = n;
	const int ldc = m;
	std::vector<double> c( matrixA.dim1() * matrixB.dim2() );
	
	dgemm( &transa, &transb, &m, &n, &k, &alpha, &a[0], &lda, &b[0], &ldb, &beta, &c[0], &ldc );

	for( int i = 0; i < matrixB.dim2(); i++ ) {
    for( int j = 0; j < matrixA.dim1(); j++ ) {
      matrixC[j][i] = c[i * matrixA.dim1() + j];
    }
  }
#else
	matrixC = matmult( matrixA, matrixB );
#endif
}

/**
 * We assume input matrix is square and symmetric.
 */
void FindEigenvalues( const TNT::Array2D<double>& matrix, std::vector<double>& values, TNT::Array2D<double>& vectors ) {
	JAMA::Eigenvalue<double> decomp( matrix );
    
    TNT::Array1D<double> eval( values.size(), 0.0 );
	decomp.getRealEigenvalues( eval );
    
	decomp.getV( vectors );
    
    for( size_t i = 0; i < values.size(); i++ ) values[i] = eval[i];
}
