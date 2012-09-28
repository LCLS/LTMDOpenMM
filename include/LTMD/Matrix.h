#ifndef LTMD_MATRIX_H_
#define LTMD_MATRIX_H_

#include <vector>
#include <algorithm>

struct Matrix{
    size_t Width, Height;
    std::vector<double> Data;

	Matrix() : Width( 0 ), Height( 0 ) {

	}
    
    Matrix( const size_t width, const size_t height ) : Width( width ), Height( height ){
        Data.resize( width * height );
    }

	Matrix( const Matrix& other ) : Width( other.Width ), Height( other.Height ){
		std::copy( other.Data.begin(), other.Data.end(), Data.begin() );
	}

	Matrix& operator=( const Matrix& other ){
		if( this != &other ){
			Width = other.Width;
			Height = other.Height;
			std::copy( other.Data.begin(), other.Data.end(), Data.begin() );
		}

		return *this;
	}
    
    double& operator()( const size_t x, const size_t y ){
        return Data[x * Width + y];
    }
    
    const double operator()( const size_t x, const size_t y ) const{
        return Data[x * Width + y];
    }
};

#endif //LTMD_MATRIX_H_
