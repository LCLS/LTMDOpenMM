#ifndef OPENMM_LTMD_MATH_H_
#define OPENMM_LTMD_MATH_H_

#include <vector>
#include "LTMD/Matrix.h"

#include "jama_eig.h"

void MatrixMultiply( const Matrix& a, const bool transposeA, const Matrix& b, const bool transposeB, Matrix& c );
bool FindEigenvalues( const Matrix& matrix, std::vector<double>& values, Matrix& vectors );

#endif // OPENMM_LTMD_MATH_H_
