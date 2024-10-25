// -*- C++ -*-
#include "../../simd.h"
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

    {
        bool masks[simd_mask::SimdLen]{false, true, false, true};
        simd_mask mask{masks};
        std::cout << "Print Simd Mask\n" ;
        std::cout << mask << "\n\n";
        std::cout << "Print Simd Mask count\n" ;
        std::cout << mask.count() << "\n\n";
        std::cout << "Print Simd Mask AND\n" ;
        std::cout << mask.AND() << "\n\n";
        std::cout << "Print Simd Mask OR\n" ;
        std::cout << mask.OR() << "\n\n";
    }

    {
        simd_mask::simd_typef masks = _mm_setr_ps(-1.0f, 0.0f, 1.0f, 0.01f);
        simd_mask mask{masks};
        std::cout << "Print Simd Mask\n" ;
        std::cout << mask << "\n\n";
        std::cout << "Print Simd Mask count\n" ;
        std::cout << mask.count() << "\n\n";
        std::cout << "Print Simd Mask AND\n" ;
        std::cout << mask.AND() << "\n\n";
        std::cout << "Print Simd Mask OR\n" ;
        std::cout << mask.OR() << "\n\n";
    }

    {
        std::cout << "\n\nBefore insert\n" ;
        bool masks[simd_mask::SimdLen]{false, false, false, true};
        simd_mask mask{masks};
        std::cout << "Print Simd Mask\n" ;
        std::cout << mask << "\n\n";
        std::cout << "Print Simd Mask count\n" ;
        std::cout << mask.count() << "\n\n";
        std::cout << "Print Simd Mask AND\n" ;
        std::cout << mask.AND() << "\n\n";
        std::cout << "Print Simd Mask OR\n" ;
        std::cout << mask.OR() << "\n\n";
        std::cout << "After insert true at index 2\n" ;
        mask.insert(2, true);
        std::cout << "Print Simd Mask\n" ;
        std::cout << mask << "\n\n";
        std::cout << "Print Simd Mask count\n" ;
        std::cout << mask.count() << "\n\n";
        std::cout << "Print Simd Mask AND\n" ;
        std::cout << mask.AND() << "\n\n";
        std::cout << "Print Simd Mask OR\n" ;
        std::cout << mask.OR() << "\n\n";
        std::cout << "After insert false at index 2\n" ;
        mask.insert(2, false);
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
