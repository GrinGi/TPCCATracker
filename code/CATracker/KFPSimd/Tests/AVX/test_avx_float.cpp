// -*- C++ -*-
#include "../../simd.h"
#include <iostream>

using KFP::SIMD::simd_float;
using KFP::SIMD::simd_int;

int main()
{
    simd_float simd_f{ 0.0f };

    std::cout << "Print simd_float{0.0f}\n" ;
    std::cout << simd_f << "\n\n";
    std::cout << "Print simd_float{0.0f} + 5\n" ;
    std::cout << simd_f + 5 << "\n\n";
    std::cout << "Print simd_float{0.0f} + simd_float{5}\n" ;
    std::cout << (simd_f + simd_float{ 5 }) << "\n\n";
    std::cout << "Print sqrt of simd_float{0.0f} + simd_float{5}\n" ;
    std::cout << sqrt(simd_f + simd_float{ 5 }) << "\n\n";
    std::cout << "Print rsqrt of simd_float{0.0f} + simd_float{5}\n" ;
    std::cout << rsqrt(simd_f + simd_float{ 5 }) << "\n\n";
    std::cout << "Print log of simd_float{0.0f} + simd_float{5}\n" ;
    std::cout << log(simd_f + simd_float{ 5 }) << "\n\n";
    std::cout << "Print power 3 of simd_float{0.0f} + simd_float{5}\n" ;
    std::cout << pow(simd_f + simd_float{ 5 }, 3) << "\n\n";

    simd_float::value_type f12345678[simd_float::SimdLen]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
std::cout << "Print load_partial(9) of {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}\n";
std::cout << simd_float{}.load_partial(9, f12345678) << "\n\n";
std::cout << "Print load_partial(9) of {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f} + simd_float{5.0f}\n";
std::cout << simd_float{}.load_partial(9, f12345678) + simd_float{5.0f} << "\n\n";

std::cout << "Print load_partial(8) of {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}\n";
std::cout << simd_float{}.load_partial(8, f12345678) << "\n\n";
std::cout << "Print load_partial(8) of {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f} + simd_float{5.0f}\n";
std::cout << simd_float{}.load_partial(8, f12345678) + simd_float{5.0f} << "\n\n";

std::cout << "Print load_partial(7) of {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}\n";
std::cout << simd_float{}.load_partial(7, f12345678) << "\n\n";
std::cout << "Print load_partial(7) of {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f} + simd_float{5.0f}\n";
std::cout << simd_float{}.load_partial(7, f12345678) + simd_float{5.0f} << "\n\n";

std::cout << "Print load_partial(6) of {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}\n";
std::cout << simd_float{}.load_partial(6, f12345678) << "\n\n";
std::cout << "Print load_partial(6) of {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f} + simd_float{5.0f}\n";
std::cout << simd_float{}.load_partial(6, f12345678) + simd_float{5.0f} << "\n\n";

std::cout << "Print simd_float to simd_int cast\n";
std::cout << simd_float{}.load_partial(9, f12345678) << "\n\n";
std::cout << "Print simd_float to simd_int cast\n";
std::cout << simd_int{simd_float{}.load_partial(9, f12345678)} << "\n\n";
std::cout << "Print simd_float to simd_int cast + simd_float{0.5f}\n";
std::cout << simd_int{simd_float{}.load_partial(9, f12345678) + simd_float{0.5f}} << "\n\n";
std::cout << "Print simd_float to simd_int cast + simd_float{0.5f} after rounding\n";
std::cout << simd_int{round(simd_float{}.load_partial(9, f12345678) + simd_float{0.5f})} << "\n\n";
}
