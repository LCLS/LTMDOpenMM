#ifndef LTMD_MATRIX_H_
#define LTMD_MATRIX_H_

#include <cassert>
#include <vector>
#include <iostream>
#include <algorithm>

struct Matrix {
	size_t Rows, Columns;
	std::vector<double> Data;

	Matrix( const size_t rows = 0, const size_t Columns = 0 );
	~Matrix();

	Matrix &operator=( const Matrix &other );

	void Print() const;

	double &operator()( const size_t row, const size_t col );
	double operator()( const size_t row, const size_t col ) const;
};

#endif //LTMD_MATRIX_H_
