#include "LTMD/Matrix.h"

#include <cstdio>

Matrix::Matrix( const size_t rows, const size_t columns ) : Rows( rows ), Columns( columns ) {
	Data.resize( rows * columns );
}

Matrix::~Matrix() {

}

Matrix &Matrix::operator=( const Matrix &other ) {
	if( this != &other ) {
		Rows = other.Rows;
		Columns = other.Columns;
		Data.resize( Rows * Columns );
		for( size_t i = 0; i < Data.size(); i++ ) {
			Data[i] = other.Data[i];
		}
	}

	return *this;
}

void Matrix::Print() const {
	for( size_t row = 0; row < Rows; row++ ) {
		for( size_t col = 0; col < Columns; col++ ) {
			if( col != Columns - 1 ) {
				printf( "%07.3f, ", ( *this )( row, col ) );
			} else {
				printf( "%07.3f\n", ( *this )( row, col ) );
			}
		}
	}
}

double &Matrix::operator()( const size_t row, const size_t col ) {
	assert( row < Rows && col < Columns );
	return Data[col * Rows + row];
}

double Matrix::operator()( const size_t row, const size_t col ) const {
	assert( row < Rows && col < Columns );
	return Data[col * Rows + row];
}
