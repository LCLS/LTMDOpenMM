#ifndef OPENMM_LTMD_MATH_H_
#define OPENMM_LTMD_MATH_H_

#include <vector>
#include "jama_eig.h"

void MatrixMultiply( const TNT::Array2D<double>& a, const TNT::Array2D<double>& b, TNT::Array2D<double>& c );
void FindEigenvalues( const TNT::Array2D<double>& matrix, std::vector<double>& values, TNT::Array2D<double>& vectors );

#endif // OPENMM_LTMD_MATH_H_
