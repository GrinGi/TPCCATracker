// -*- C++ -*-
#include "../simd.h"
#include <iostream>

using KFP::SIMD::simd_index;

int main()
{
    {
        simd_index index;
        std::cout << "Print Simd Index\n" ;
        std::cout << index << "\n\n";
        std::cout << "Print Simd Index op+ 5\n" ;
        std::cout << index + 5 << "\n\n";
        std::cout << "Print Simd Index seq\n" ;
        std::cout << simd_index(3) << "\n\n";
    }

    {
        simd_index index{5};
        std::cout << "Print Simd Index\n" ;
        std::cout << index << "\n\n";
        std::cout << "Print Simd Index op/ 5\n" ;
        std::cout << index / 5 << "\n\n";
        std::cout << "Print Simd Index seq\n" ;
        std::cout << simd_index(5) << "\n\n";
    }

}
