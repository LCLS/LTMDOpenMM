#include "LTMD/Matrix.h"

#include <cstdio>

Matrix::Matrix( const size_t width, const size_t height ) : Width( width ), Height( height ){
    Data.resize( width * height );
}

Matrix::~Matrix(){
    
}

Matrix& Matrix::operator=( const Matrix& other ){
    if( this != &other ){
        Width = other.Width;
        Height = other.Height;
        Data.resize( Width * Height );
        for( size_t i = 0; i < Data.size(); i++ ){
            Data[i] = other.Data[i];
        }
    }
    
    return *this;
}

void Matrix::Print() const {
    for( size_t row = 0; row < Width; row++ ){
        for( size_t col = 0; col < Height; col++ ){
            if( col != Height-1 ){
                printf( "%07.3f, ", (*this)(row, col));
            }else{
                printf( "%07.3f\n", (*this)(row, col));
            }
        }
    }
}

double& Matrix::operator()( const size_t row, const size_t col ){
    assert( row < Width && col < Height );
    return Data[col * Width + row];
}

const double Matrix::operator()( const size_t row, const size_t col ) const{
    assert( row < Width && col < Height );
    return Data[col * Width + row];
}
