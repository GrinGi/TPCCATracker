// -*- C++ -*-
#include "../simd.h"
#include <iostream>

using KFP::SIMD::simd_mask;

int main()
{
    {
        simd_mask mask;
        std::cout << "Print Simd Mask\n" ;
        std::cout << mask << "\n\n";
        std::cout << "Print Simd Mask count\n" ;
        std::cout << mask.count() << "\n\n";
        std::cout << "Print Simd Mask AND\n" ;
        std::cout << mask.AND() << "\n\n";
        std::cout << "Print Simd Mask OR\n" ;
        std::cout << mask.OR() << "\n\n";
        std::cout << "Print Simd Mask == Simd Mask{false}\n" ;
        std::cout << (mask == simd_mask{false}) << "\n\n";
        std::cout << "Print !Simd Mask == Simd Mask{true}\n" ;
        std::cout << ((!mask) == simd_mask{true}) << "\n\n";
        std::cout << "Print !Simd Mask == Simd Mask{false}\n" ;
        std::cout << ((!mask) == simd_mask{}) << "\n\n";
    }

    {
        simd_mask mask{true};
        std::cout << "Print Simd Mask\n" ;
        std::cout << mask << "\n\n";
        std::cout << "Print Simd Mask count\n" ;
        std::cout << mask.count() << "\n\n";
        std::cout << "Print Simd Mask AND\n" ;
        std::cout << mask.AND() << "\n\n";
        std::cout << "Print Simd Mask OR\n" ;
        std::cout << mask.OR() << "\n\n";
    }

}
