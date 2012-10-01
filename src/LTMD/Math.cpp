#include "LTMD/Math.h"

#ifdef INTEL_MKL
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#else
extern "C" void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
extern "C" void dsyev_(char*, char*, int*, double*, int*, double*, double*, int*, int*);
#endif

#include <vector>

void MatrixMultiply( const Matrix& matrixA, const bool transposeA, const Matrix& matrixB, const bool transposeB, Matrix& matrixC ) {
    char transa = transposeA ? 'T' : 'N';
    char transb = transposeB ? 'T' : 'N';
    
	int m = transposeA ? matrixA.Height : matrixA.Width;
    int n = transposeB ? matrixB.Width : matrixB.Height;
    int k = transposeA ? matrixA.Width : matrixA.Height;
    
	double alpha = 1.0, beta = 0.0;
    
	int lda = transposeA ? k : m;
    int ldb = transposeB ? n : k;
    int ldc = m;

#ifdef INTEL_MKL
	dgemm( &transa, &transb, &m, &n, &k, &alpha, &matrixA.Data[0], &lda, &matrixB.Data[0], &ldb, &beta, &matrixC.Data[0], &ldc );
#else
    dgemm_( &transa, &transb, &m, &n, &k, &alpha, (double*)&matrixA.Data[0], &lda, (double*)&matrixB.Data[0], &ldb, &beta, &matrixC.Data[0], &ldc );
#endif
}

/**
 * We assume input matrix is square and symmetric.
 */

/*
 double* work;
double w[N];
dsyev( "Vectors", "Upper", &n, a, &lda, w, &wkopt, &lwork, &info );
 
dsyev( "Vectors", "Upper", &n, a, &lda, w, work, &lwork, &info );
 */
void FindEigenvalues( const Matrix& matrix, std::vector<double>& values, Matrix& vectors ) {
    vectors = matrix;
    int n = matrix.Width, lda = matrix.Width, lwork = -1, info = 0;
    
    double wkopt = 0;
    dsyev_("V", "U", &n, (double*)&vectors.Data[0], &lda, &values[0], &wkopt, &lwork, &info);
    
    if(info == 0){
        lwork = (int)wkopt;
        double *wrkSp = new double[lwork];
        dsyev_("V", "U", &n, (double*)&vectors.Data[0], &lda, &values[0], wrkSp, &lwork, &info);
    }
}
