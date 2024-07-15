// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_DATA_H
#define SIMD_DATA_H

#include "simd_macros.h"
#include "simd_tag.h"

#include <x86intrin.h>

namespace KFP
{
namespace SIMD
{

template<typename T, Tag tag>
struct SimdData
{
    static_assert((tag == Tag::Scalar), "[Error] (SimdData): Invalid use of primary template of SimdData meant for Scalar tag.") ;
    typedef T value_type;
    typedef T simd_type;
    T simd_ ;
};

#if defined(__KFP_SIMD__AVX)
template<>
struct SimdData<int, Tag::AVX>
{
    typedef int value_type;
    typedef __m256i simd_type;
    __m256i simd_ ;
};

template<>
struct SimdData<float, Tag::AVX>
{
    typedef float value_type;
    typedef __m256 simd_type;
    __m256 data_ ;
};
#elif defined(__KFP_SIMD__SSE)
template<>
struct SimdData<int, Tag::SSE>
{
    typedef int value_type;
    typedef __m128i simd_type;
    __m128i simd_ ;
    // int __KFP_SIMD__ATTR_ALIGN(
    //     __KFP_SIMD__Size_Int) simd_[__KFP_SIMD__Len_Int]{}; // Data array
    // __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) int simd_[__KFP_SIMD__Len_Int]{}; // Data array
};

template<>
struct SimdData<float, Tag::SSE>
{
    typedef float value_type;
    typedef __m128 simd_type;
    __m128 simd_ ;
    // float __KFP_SIMD__ATTR_ALIGN(
    //     __KFP_SIMD__Size_Float) simd_[__KFP_SIMD__Len_Float]{}; // Data array
    // __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Float) float simd_[__KFP_SIMD__Len_Float]{}; // Data array
};
#endif

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_DATA_H
