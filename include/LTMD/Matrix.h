#ifndef LTMD_MATRIX_H_
#define LTMD_MATRIX_H_

#include <vector>

struct Matrix{
    const size_t Width, Height;
    std::vector<double> Data;
    
    Matrix( const size_t width, const size_t height ) : Width( width ), Height( height ){
        Data.resize( width * height );
    }
    
    double& operator()( const size_t x, const size_t y ){
        return Data[x * Width + y];
    }
    
    const double operator()( const size_t x, const size_t y ) const{
        return Data[x * Width + y];
    }
};

#endif //LTMD_MATRIX_H_