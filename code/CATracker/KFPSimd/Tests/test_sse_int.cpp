// -*- C++ -*-
#include "../simd.h"
#include <iostream>

using KFP::SIMD::simd_mask;
using KFP::SIMD::simd_int;
using KFP::SIMD::simd_float;

int main()
{
    simd_int simd_i{ 0 };

    std::cout << "Print simd_int{0}\n" ;
    std::cout << simd_i << "\n\n";
    std::cout << "Print simd_int{0} + 5\n" ;
    std::cout << simd_i + 5 << "\n\n";
    std::cout << "Print simd_int{0} + simd_int{5}\n" ;
    std::cout << (simd_i + simd_int{ 5 }) << "\n\n";

    std::cout << "Print sqrt of simd_int{0} + simd_int{5}\n" ;
    std::cout << sqrt(simd_i + simd_int{ 5 }) << "\n\n";
    std::cout << "Print rsqrt of simd_int{0} + simd_int{5}\n" ;
    std::cout << rsqrt(simd_i + simd_int{ 5 }) << "\n\n";
    std::cout << "Print log of simd_int{0} + simd_int{5}\n" ;
    std::cout << log(simd_i + simd_int{ 5 }) << "\n\n";
    std::cout << "Print power 3 of simd_int{0} + simd_int{5}\n" ;
    std::cout << pow(simd_i + simd_int{ 5 }, 3) << "\n\n";

    simd_int::value_type i1234[simd_int::SimdLen]{1, 2, 3, 4};
    std::cout << "Print load_partial(5) of {1, 2, 3, 4}\n" ;
    std::cout << simd_int{}.load_partial(5, i1234) << "\n\n";
    std::cout << "Print load_partial(5) of {1, 2, 3, 4} + simd_int{5}\n" ;
    std::cout << simd_int{}.load_partial(5, i1234) + simd_int{5} << "\n\n";

    std::cout << "Print load_partial(4) of {1, 2, 3, 4}\n" ;
    std::cout << simd_int{}.load_partial(4, i1234) << "\n\n";
    std::cout << "Print load_partial(4) of {1, 2, 3, 4} + simd_int{5}\n" ;
    std::cout << simd_int{}.load_partial(4, i1234) + simd_int{5} << "\n\n";

    std::cout << "Print load_partial(3) of {1, 2, 3, 4}\n" ;
    std::cout << simd_int{}.load_partial(3, i1234) << "\n\n";
    std::cout << "Print load_partial(3) of {1, 2, 3, 4} + simd_int{5}\n" ;
    std::cout << simd_int{}.load_partial(3, i1234) + simd_int{5} << "\n\n";

    std::cout << "Print load_partial(2) of {1, 2, 3, 4}\n" ;
    std::cout << simd_int{}.load_partial(2, i1234) << "\n\n";
    std::cout << "Print load_partial(2) of {1, 2, 3, 4} + simd_int{5}\n" ;
    std::cout << simd_int{}.load_partial(2, i1234) + simd_int{5} << "\n\n";

    std::cout << "Print simd_int to simd_float cast\n" ;
    std::cout << simd_int{}.load_partial(5, i1234) << "\n\n";
    std::cout << "Print simd_int to simd_float cast\n" ;
    std::cout << simd_float{simd_int{}.load_partial(5, i1234)} << "\n\n";
    std::cout << "Print simd_int to simd_float cast + simd_float{0.5f}\n" ;
    std::cout << simd_float{simd_int{}.load_partial(5, i1234)} + simd_float{0.5f} << "\n\n";
}
