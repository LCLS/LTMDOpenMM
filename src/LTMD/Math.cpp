#include "LTMD/Math.h"

#ifdef INTEL_MKL
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#else
extern "C" void dgemm_( char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int * );

extern "C" double dlamch_( char * );
extern "C" void dsyevr_( char *, char *, char *, int *, double *, int *, double *, double *, int *, int *, double *, int *, double *, double *, int *, int *, double *, int *, int *, int *, int * );

#endif

#include <vector>

void MatrixMultiply( const Matrix &matrixA, const bool transposeA, const Matrix &matrixB, const bool transposeB, Matrix &matrixC ) {
	char transa = transposeA ? 'T' : 'N';
	char transb = transposeB ? 'T' : 'N';

	int m = transposeA ? matrixA.Columns : matrixA.Rows;
	int n = transposeB ? matrixB.Rows : matrixB.Columns;
	int k = transposeA ? matrixA.Rows : matrixA.Columns;

	double alpha = 1.0, beta = 0.0;

	int lda = transposeA ? k : m;
	int ldb = transposeB ? n : k;
	int ldc = m;

#ifdef INTEL_MKL
	dgemm( &transa, &transb, &m, &n, &k, &alpha, &matrixA.Data[0], &lda, &matrixB.Data[0], &ldb, &beta, &matrixC.Data[0], &ldc );
#else
	dgemm_( &transa, &transb, &m, &n, &k, &alpha, ( double * )&matrixA.Data[0], &lda, ( double * )&matrixB.Data[0], &ldb, &beta, &matrixC.Data[0], &ldc );
#endif
}

bool FindEigenvalues( const Matrix &matrix, std::vector<double> &values, Matrix &vectors ) {
	Matrix temp = matrix;
	vectors = matrix;

	int m = 0, n = matrix.Rows, lda = n, ldz = n;

	int lwork = 26 * n, liwork = 10 * n;
	std::vector<int> isuppz( 2 * n );
	std::vector<int> iwork( 10 * n );
	std::vector<double> wrkSp( 26 * n );

	int il = 1, iu = 1;
	double vl = 1.0, vu = 1.0;

	int info = 0;

	double abstol = dlamch_( "s" );
	dsyevr_( "V", "A", "U", &n, &temp.Data[0], &lda, &vl, &vu, &il, &iu, &abstol, &m, &values[0], &vectors.Data[0], &ldz, &isuppz[0], &wrkSp[0], &lwork, &iwork[0], &liwork, &info );

	return ( info == 0 );
}

/*


    return info;
*/