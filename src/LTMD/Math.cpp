#include "LTMD/Math.h"

#ifdef INTEL_MKL
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#else
extern "C" void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
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
void FindEigenvalues( const Matrix& matrix, std::vector<double>& values, Matrix& vectors ) {
    TNT::Array2D<double> mat( matrix.Width, matrix.Height );
    for( size_t i = 0; i < mat.dim1(); i++ ){
        for( size_t j = 0; j < mat.dim2(); j++ ){
            mat[i][j] = matrix( i, j );
        }
    }
    
	JAMA::Eigenvalue<double> decomp( mat );
    
    TNT::Array1D<double> eval( values.size(), 0.0 );
	decomp.getRealEigenvalues( eval );
    
    TNT::Array2D<double> evec( vectors.Width, vectors.Height );
	decomp.getV( evec );
    
    for( size_t i = 0; i < evec.dim1(); i++ ){
        values[i] = eval[i];
        for( size_t j = 0; j < evec.dim2(); j++ ){
            vectors( i, j ) = evec[i][j];
        }
    }
}
