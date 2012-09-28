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
            Data.resize( Width * Height );
            for( size_t i = 0; i < Data.size(); i++ ){
                Data[i] = other.Data[i];
            }
		}

		return *this;
	}
    
    double& operator()( const size_t row, const size_t col ){
        return Data[col * Height + row];
    }
    
    const double operator()( const size_t row, const size_t col ) const{
        return Data[col * Height + row];
    }
};

#endif //LTMD_MATRIX_H_
