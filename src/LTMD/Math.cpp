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
	int lda = k;
	int ldb = n;
	int ldc = m;
	std::vector<double> a( m * k );
	std::vector<double> b( k * n );
	std::vector<double> c( m * n );

	for( int i = 0; i < m; i++ ) {
		for( int j = 0; j < k; j++ ) {
			a[i * k + j] = matrixA[i][j];
		}
	}

	for( int i = 0; i < n; i++ ) {
		for( int j = 0; j < k; j++ ) {
			b[i * n + j] = matrixB[i][j];
		}
	}

	dgemm( &transa, &transb, &m, &n, &k, &alpha, &a[0], &lda, &b[0], &ldb, &beta, &c[0], &ldc );

	for( int i = 0; i < m; i++ ) {
		for( int j = 0; j < n; j++ ) {
			matrixC[j][i] = c[i * n + j];
		}
	}
#else
	matrixC = matmult( matrixA, matrixB );
#endif
}

void FindEigenvalues( const TNT::Array2D<double>& matrix, TNT::Array1D<double>& values, TNT::Array2D<double>& vectors ) {
#ifdef INTEL_MKL
	const int dim = matrix.dim1();

	const char cmach = 'S';
	const double abstol = dlamch( &cmach );

	std::vector<int> isuppz( 2 * dim ), iwork( 10 * dim );
	std::vector<double> wrkSp( 26 * dim );
	const char jobz = 'V', range = 'A', uplo = 'U';
	const int n = dim, lda = dim;
	const double vl = 1.0, vu = 1.0;
	const int il = 1, iu = 1;
	const int ldz = dim, lwork = 26 * dim, liwork = 10 * dim;
	int info = 0, m = 0;

	std::vector<double> iMatrix( dim * dim ), oValue( dim ), oVector( dim * dim );
	for( int i = 0; i < dim; i++ ) {
		for( int j = 0; j < dim; j++ ) {
			iMatrix[i * dim + j] = matrix[i][j];
		}
	}

	dsyevr( &jobz, &range, &uplo, &n, &iMatrix[0], &lda, &vl, &vu, &il, &iu, &abstol, &m, &oValue[0], &oVector[0], &ldz, &isuppz[0], &wrkSp[0], &lwork, &iwork[0], &liwork, &info );

	for( int i = 0; i < dim; i++ ) {
		values[i] = oValue[i];
		for( int j = 0; j < dim; j++ ) {
			vectors[i][j] = oVector[i + j * dim];
		}
	}
#else
	JAMA::Eigenvalue<double> decomp( matrix );
	decomp.getRealEigenvalues( values );
	decomp.getV( vectors );
#endif
}
