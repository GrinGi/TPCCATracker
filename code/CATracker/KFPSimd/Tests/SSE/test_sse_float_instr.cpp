// -*- C++ -*-
#include "../../simd.h"
#include <iostream>

using KFP::SIMD::simd_float;
using KFP::SIMD::simd_int;

simd_float getABS(simd_float v)
{
    return abs(v);
}

int main()
{
    // alignas(16) simd_float::value_type f1234[simd_float::SimdLen]{-1.0f, 2.0f, -3.0f, 4.0f};
    std::cout << simd_float::SimdLen << "\n\n";
    // simd_float val_simd{ f1234 };
    // simd_float val_simd{ _mm_set1_ps(10.0f) };
    __m128 val_simd{ _mm_set1_ps(10.0f) };
    // std::cout << "The value of val_simd:\n" ;
    // std::cout << val_simd << "\n\n";

    // simd_float val_simd_abs{ getABS(val_simd) };
    // std::cout << "The value of abs val_simd:\n" ;
    // std::cout << val_simd_abs << "\n\n";
}
