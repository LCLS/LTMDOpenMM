#include "LTMD/Math.h"

#ifdef INTEL_MKL
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#endif
#include "tnt_array2d_utils.h"
//#endif

#include <vector>

void MatrixMultiply( const TNT::Array2D<double>& matrixA, const TNT::Array2D<double>& matrixB, TNT::Array2D<double>& matrixC ) {
#ifdef INTEL_MKL
	const int m = matrixA.dim1();
	const int k = matrixA.dim2();
	const int n = matrixB.dim2();

        std::vector<double> a( m * k );

	for( int i = 0; i < matrixA.dim1(); i++ )
	  {
	    for( int j = 0; j < matrixA.dim2(); j++ )
	      {
		a[i * matrixA.dim2() + j] = matrixA[i][j];
	      }
	  }

        std::vector<double> b( k * n);

	for( int i = 0; i < matrixB.dim1(); i++ )
	  {
	    for( int j = 0; j < matrixB.dim2(); j++ )
	      {
		b[i * matrixB.dim2() + j] = matrixB[i][j];
	      }
	  }

	const char transa = 'N';
	const char transb = 'N';
	const double alpha = 1.0;
	const double beta = 0.0;
	const int lda = m;
	const int ldb = k;
	const int ldc = m;
	std::vector<double> c( m * n );
	
	dgemm( &transa, &transb, &m, &n, &k, &alpha, &a[0], &lda, &b[0], &ldb, &beta, &c[0], &ldc );
	std::cout << "completed call to dgemm" << std::endl;

	std::cout << m << " " << n << std::endl;

	for( int i = 0; i < m; i++ ) {
		for( int j = 0; j < n; j++ ) {
			matrixC[i][j] = c[i * n + j];
		}
	}
#else
	matrixC = matmult( matrixA, matrixB );
#endif
}

/**
 * We assume input matrix is square and symmetric.
 */
void FindEigenvalues( const TNT::Array2D<double>& matrix, TNT::Array1D<double>& values, TNT::Array2D<double>& vectors ) {
#ifdef INTEL_MKL
  std::cout << "calling mkl diagonalize" << std::endl;
	const int dim = matrix.dim1();

	const char cmach = 'S';
	const double abstol = dlamch( &cmach ); 
	
	const char jobz = 'V'; // compute eigenvectors and eigenvalues
	const char range = 'A'; // compute all eigenvalues
	const char uplo = 'U'; // use upper triangular part
	const int n = dim;
	const int lda = dim;
	const int ldz = dim;

	const int lwork = 26 * n; // workspace size suggested by manual
	const int liwork = 10 * n; //  i work size suggested by manual

	// input matrices
	std::vector<double> a(lda * n); // matrix
	std::vector<double> work(lwork); // workspace
	std::vector<double> iwork(liwork); // iwork workspace

	// not referenced
	double vl = 0.0;
	double vu = 0.0;
	double il = 0.0;
	double iu = 0.0;

	// output parameters
	int m = n; // number eigenpairs found, should equal n
	std::vector<double> w(m); // stores eigenvalues
	std::vector<double> z(ldz*m); // stores eigenvectors
	std::vector<int> isuppz(2 * m); // gives indices bounding non-zero regions of eigenvectors
	
	for( int i = 0; i < lda; i++ ) {
		for( int j = 0; j < n; j++ ) {
			a[i * lda + j] = matrix[i][j];
		}
	}
	
	double *wPointer = &w[0];
	double *zPointer = &z[0];

	int info = LAPACKE_dsyevr( LAPACK_ROW_MAJOR, jobz, range, uplo, n, &a[0], lda, vl, vu, il, iu, abstol, &m, wPointer, zPointer, ldz, &isuppz[0]);

	if( wPointer != &w[0] ) std::cout << "WDiffers" << std::endl;
	if( zPointer != &z[0] ) std::cout << "ZDiffers" << std::endl;

	std::cout << "info " << info << std::endl;
	
	for( int i = 0; i < dim; i++ ) {
		values[i] = w[i];
		for( int j = 0; j < dim; j++ ) {
			vectors[i][j] = z[i * dim + j];
		}
	}
#else
	JAMA::Eigenvalue<double> decomp( matrix );
	decomp.getRealEigenvalues( values );
	decomp.getV( vectors );
#endif
}
