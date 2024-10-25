// -*- C++ -*-

#include "../Base/simd_allocate.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <x86intrin.h>

inline bool
isAligned(const void * ptr, std::uintptr_t alignment) noexcept {
    auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
    std::size_t aligned = (iptr % alignment) ;
    return (aligned == 0) ;
}

template<typename T>
inline bool
isAligned_NewExp(std::uintptr_t alignment = alignof(T)) noexcept {
    for (std::size_t idx{0} ; idx < 10000 ; ++idx) {
        T* ptr = new T[idx] ;
        const bool aligned{ isAligned(ptr, alignment) };
        delete[] ptr ;
        if(not aligned) return false;
    }
    return true ;
}

template<typename T>
inline bool
isAligned_NewOp(std::uintptr_t alignment = alignof(T)) noexcept {
    for (std::size_t idx{0} ; idx < 10000 ; ++idx) {
        T* ptr = static_cast<T*>( ::operator new(idx * sizeof(T)) );
        const bool aligned{ isAligned(ptr, alignment) };
        ::operator delete(ptr) ;
        if(not aligned) return false;
    }
    return true ;
}

template<typename T>
inline bool
isAligned_alignedAlocate(std::uintptr_t alignment = alignof(T)) noexcept {
    for (std::size_t idx{0} ; idx < 10000 ; ++idx) {
        T* ptr = static_cast<T*>( KFP::SIMD::alignedAllocate(idx * sizeof(T), alignment) );
        const bool aligned{ isAligned(ptr, alignment) };
        KFP::SIMD::alignedDeallocate(ptr) ;
        if(not aligned) return false;
    }
    return true ;
}

const std::size_t AVX512_ALIGN = 64 ;
const std::size_t AVX_ALIGN = 32 ;
const std::size_t SSE_ALIGN = 16 ;
const std::size_t DOUBLE_ALIGN = 8 ;
const std::size_t FLOAT_ALIGN = 4 ;
const std::size_t INT_ALIGN = 4 ;

int main()
{

    std::cout << std::boolalpha ;

    std::cout << "double is aligned on new exp to : " << DOUBLE_ALIGN << " ? " << isAligned_NewExp<double>(DOUBLE_ALIGN) << std::endl ;
    std::cout << "double is aligned on new exp to : " << FLOAT_ALIGN << " ? " << isAligned_NewExp<double>(FLOAT_ALIGN) << std::endl ;
    std::cout << "double is aligned on new exp to : " << INT_ALIGN << " ? " << isAligned_NewExp<double>(INT_ALIGN) << std::endl ;
    std::cout << "double is aligned on new exp to : " << SSE_ALIGN << " ? " << isAligned_NewExp<double>(SSE_ALIGN) << std::endl ;
    std::cout << "double is aligned on new exp to : " << AVX_ALIGN << " ? " << isAligned_NewExp<double>(AVX_ALIGN) << std::endl ;
    std::cout << "double is aligned on new exp to : " << AVX512_ALIGN << " ? " << isAligned_NewExp<double>(AVX512_ALIGN) << std::endl ;

    std::cout << "double is aligned on new op to : " << DOUBLE_ALIGN << " ? " << isAligned_NewOp<double>(DOUBLE_ALIGN) << std::endl ;
    std::cout << "double is aligned on new op to : " << FLOAT_ALIGN << " ? " << isAligned_NewOp<double>(FLOAT_ALIGN) << std::endl ;
    std::cout << "double is aligned on new op to : " << INT_ALIGN << " ? " << isAligned_NewOp<double>(INT_ALIGN) << std::endl ;
    std::cout << "double is aligned on new op to : " << SSE_ALIGN << " ? " << isAligned_NewOp<double>(SSE_ALIGN) << std::endl ;
    std::cout << "double is aligned on new op to : " << AVX_ALIGN << " ? " << isAligned_NewOp<double>(AVX_ALIGN) << std::endl ;
    std::cout << "double is aligned on new op to : " << AVX512_ALIGN << " ? " << isAligned_NewOp<double>(AVX512_ALIGN) << std::endl ;

    std::cout << "double is aligned on alignedAllocate to : " << DOUBLE_ALIGN << " ? " << isAligned_alignedAlocate<double>(DOUBLE_ALIGN) << std::endl ;
    std::cout << "double is aligned on alignedAllocate to : " << FLOAT_ALIGN << " ? " << isAligned_alignedAlocate<double>(FLOAT_ALIGN) << std::endl ;
    std::cout << "double is aligned on alignedAllocate to : " << INT_ALIGN << " ? " << isAligned_alignedAlocate<double>(INT_ALIGN) << std::endl ;
    std::cout << "double is aligned on alignedAllocate to : " << SSE_ALIGN << " ? " << isAligned_alignedAlocate<double>(SSE_ALIGN) << std::endl ;
    std::cout << "double is aligned on alignedAllocate to : " << AVX_ALIGN << " ? " << isAligned_alignedAlocate<double>(AVX_ALIGN) << std::endl ;
    std::cout << "double is aligned on alignedAllocate to : " << AVX512_ALIGN << " ? " << isAligned_alignedAlocate<double>(AVX512_ALIGN) << std::endl ;

    std::cout << "__m128 is aligned on new exp to : " << DOUBLE_ALIGN << " ? " << isAligned_NewExp<__m128>(DOUBLE_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new exp to : " << FLOAT_ALIGN << " ? " << isAligned_NewExp<__m128>(FLOAT_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new exp to : " << INT_ALIGN << " ? " << isAligned_NewExp<__m128>(INT_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new exp to : " << SSE_ALIGN << " ? " << isAligned_NewExp<__m128>(SSE_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new exp to : " << AVX_ALIGN << " ? " << isAligned_NewExp<__m128>(AVX_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new exp to : " << AVX512_ALIGN << " ? " << isAligned_NewExp<__m128>(AVX512_ALIGN) << std::endl ;

    std::cout << "__m128 is aligned on new op to : " << DOUBLE_ALIGN << " ? " << isAligned_NewOp<__m128>(DOUBLE_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new op to : " << FLOAT_ALIGN << " ? " << isAligned_NewOp<__m128>(FLOAT_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new op to : " << INT_ALIGN << " ? " << isAligned_NewOp<__m128>(INT_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new op to : " << SSE_ALIGN << " ? " << isAligned_NewOp<__m128>(SSE_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new op to : " << AVX_ALIGN << " ? " << isAligned_NewOp<__m128>(AVX_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on new op to : " << AVX512_ALIGN << " ? " << isAligned_NewOp<__m128>(AVX512_ALIGN) << std::endl ;

    std::cout << "__m128 is aligned on alignedAllocate to : " << DOUBLE_ALIGN << " ? " << isAligned_alignedAlocate<__m128>(DOUBLE_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on alignedAllocate to : " << FLOAT_ALIGN << " ? " << isAligned_alignedAlocate<__m128>(FLOAT_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on alignedAllocate to : " << INT_ALIGN << " ? " << isAligned_alignedAlocate<__m128>(INT_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on alignedAllocate to : " << SSE_ALIGN << " ? " << isAligned_alignedAlocate<__m128>(SSE_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on alignedAllocate to : " << AVX_ALIGN << " ? " << isAligned_alignedAlocate<__m128>(AVX_ALIGN) << std::endl ;
    std::cout << "__m128 is aligned on alignedAllocate to : " << AVX512_ALIGN << " ? " << isAligned_alignedAlocate<__m128>(AVX512_ALIGN) << std::endl ;

}

