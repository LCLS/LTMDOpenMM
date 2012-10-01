#ifndef LTMD_MATRIX_H_
#define LTMD_MATRIX_H_

#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>

struct Matrix{
    size_t Width, Height;
    std::vector<double> Data;

    Matrix( const size_t width = 0, const size_t height = 0 );
    ~Matrix();

	Matrix& operator=( const Matrix& other );
    
    void Print() const;
    
    double& operator()( const size_t row, const size_t col );
    const double operator()( const size_t row, const size_t col ) const;
};

#endif //LTMD_MATRIX_H_
