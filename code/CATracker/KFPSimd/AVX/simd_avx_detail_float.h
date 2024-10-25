// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_AVX_DETAIL_FLOAT_H
#define SIMD_AVX_DETAIL_FLOAT_H

#include "../Base/simd_macros.h"
#include "../Base/simd_detail.h"
#include "simd_avx_detail_int.h"
#include "simd_avx_type.h"

#include <x86intrin.h>
#include <cmath>
#include <iostream>

namespace KFP {
namespace SIMD {

namespace Detail {

// ------------------------------------------------------
// General
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE __m256 type_cast<__m256, __m256i>(const __m256i& val_simd)
{
    return _mm256_castsi256_ps(val_simd);
}
template <>
__KFP_SIMD__INLINE __m256 type_cast<__m256>(const __m256 &val_simd) {
  return val_simd;
}
template <> __KFP_SIMD__INLINE __m256 value_cast<__m256, __m256i>(const __m256i& val_simd)
{
    return _mm256_cvtepi32_ps(val_simd);
}
template <>
__KFP_SIMD__INLINE __m256 value_cast<__m256>(const __m256 &val_simd) {
  return val_simd;
}
template <> __KFP_SIMD__INLINE __m256 constant<__m256, float>(float val)
{
    return _mm256_set1_ps(val);
}
// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE __m256 load<__m256, float>(const float* val_ptr)
{
    return _mm256_loadu_ps(val_ptr);
}

template <>
__KFP_SIMD__INLINE __m256 load_a<__m256, float>(const float* val_ptr)
{
    return _mm256_load_ps(val_ptr);
}

template <>
__KFP_SIMD__INLINE void store<__m256, float>(const __m256& val_simd, float* val_ptr)
{
    _mm256_storeu_ps(val_ptr, val_simd);
}

template <>
__KFP_SIMD__INLINE void store_a<__m256, float>(const __m256& val_simd, float* val_ptr)
{
    _mm256_store_ps(val_ptr, val_simd);
}

// ------------------------------------------------------
// Logical bitwise
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE __m256 ANDBits<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_and_ps(a, b);
}
template <>
__KFP_SIMD__INLINE __m256 ORBits<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_or_ps(a, b);
}
template <>
__KFP_SIMD__INLINE __m256 XORBits<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_xor_ps(a, b);
}
// template <>
// __KFP_SIMD__INLINE SimdDataF NOTBits<SimdDataF>(const SimdDataF& a)
// {
    // return XORBits<SimdDataF>(type_cast<SimdDataF, SimdDataI>(getMask<MASK::TRUE>()), a);
// }

// ------------------------------------------------------
// Comparison
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE __m256 equal<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
}
template <>
__KFP_SIMD__INLINE __m256 notEqual<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
}
template <>
__KFP_SIMD__INLINE __m256 lessThan<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}
template <>
__KFP_SIMD__INLINE __m256 lessThanEqual<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
}
template <>
__KFP_SIMD__INLINE __m256 greaterThan<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
}
template <>
__KFP_SIMD__INLINE __m256 greaterThanEqual<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
}

// ------------------------------------------------------
// Manipulate lanes
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE __m256 select(const __m256& mask, const __m256& a, const __m256& b)
{
    return _mm256_blendv_ps(b, a, mask);
}

template <int N> __KFP_SIMD__INLINE float get(const __m256& a)
{
    float result[8];
    _mm256_storeu_ps(result, a);
    return result[N];
}

// Pavel: TODO:
// template <>
// __KFP_SIMD__INLINE float extract<int, __m256>(int index, const __m256& a)
// {
//     float result[8];
//     _mm256_storeu_ps(result, a);
//     return result[index];
// }

// Robin: Maybe this instead:
/*
template <int N> __KFP_SIMD__INLINE ValueDataF get(const SimdDataF& a)
{
    const SimdDataF result = _mm256_permute_ps(a, (N & 7));
    return _mm_cvtss_f32(_mm256_castps256_ps128(result));
}*/
template <>
__KFP_SIMD__INLINE float extract<float, __m256>(int index, const __m256& vec) {
    return _mm256_cvtss_f32(_mm256_permutevar8x32_ps(vec, _mm256_set1_epi32(index)));
}

template <>
__KFP_SIMD__INLINE void insert<__m256, float>(__m256 &val_simd, int index, float val) {
    float data[8];
    _mm256_storeu_ps(data, val_simd);
    data[index] = val;
    val_simd = _mm256_loadu_ps(data);
}

template <>
__KFP_SIMD__INLINE __m256 shiftLLanes<__m256>(size_t amount, const __m256 &val_simd) {
    __m256i tmp = _mm256_permutevar8x32_epi32(_mm256_castps_si256(val_simd), _mm256_setr_epi32((amount % 8), (1 + amount) % 8, (2 + amount) % 8, (3 + amount) % 8, (4 + amount) % 8, (5 + amount) % 8, (6 + amount) % 8, (7 + amount) % 8));
    return _mm256_castsi256_ps(tmp);
}

template <>
__KFP_SIMD__INLINE __m256 shiftRLanes<__m256>(size_t amount, const __m256 &val_simd) {
    __m256i tmp = _mm256_permutevar8x32_epi32(_mm256_castps_si256(val_simd), _mm256_setr_epi32((8 - amount) % 8, (9 - amount) % 8, (10 - amount) % 8, (11 - amount) % 8, (12 - amount) % 8, (13 - amount) % 8, (14 - amount) % 8, (15 - amount) % 8));
    return _mm256_castsi256_ps(tmp);
}

template <> __KFP_SIMD__INLINE __m256 rotate<__m256>(size_t amount, const __m256& val_simd)
{
    __m256i tmp = _mm256_permutevar8x32_epi32(_mm256_castps_si256(val_simd), _mm256_setr_epi32(amount % 8, (1 + amount) % 8, (2 + amount) % 8, (3 + amount) % 8, (4 + amount) % 8, (5 + amount) % 8, (6 + amount) % 8, (7 + amount) % 8));
    return _mm256_castsi256_ps(tmp);
}

// ------------------------------------------------------
// Basic Arithmetic
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE __m256 add<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_add_ps(a, b);
}
template <>
__KFP_SIMD__INLINE __m256 substract<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_sub_ps(a, b);
}
template <>
__KFP_SIMD__INLINE __m256 multiply<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_mul_ps(a, b);
}
template <>
__KFP_SIMD__INLINE __m256 divide<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_div_ps(a, b);
}
template <> __KFP_SIMD__INLINE __m256 minus<__m256>(const __m256& a)
{
    return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
}

template <>
__KFP_SIMD__INLINE __m256 min<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_min_ps(a, b);
}
template <>
__KFP_SIMD__INLINE __m256 max<__m256>(const __m256& a, const __m256& b)
{
    return _mm256_max_ps(a, b);
}

template <> __KFP_SIMD__INLINE __m256 sqrt<__m256>(const __m256& a)
{
    return _mm256_sqrt_ps(a);
}
template <> __KFP_SIMD__INLINE __m256 rsqrt<__m256>(const __m256& a)
{
    return _mm256_rsqrt_ps(a);
}
template <> __KFP_SIMD__INLINE __m256 abs<__m256>(const __m256& a)
{
    return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
}
template <> __KFP_SIMD__INLINE __m256 log<__m256>(const __m256& a)
{
    alignas(32) float data[8];
    _mm256_store_ps(data, a);
    for (int i = 0; i < 8; ++i) {
        data[i] = std::log(data[i]);
    }
    return _mm256_load_ps(data);
}
template <> __KFP_SIMD__INLINE SimdDataF pow<SimdDataF>(const SimdDataF& a, int exp)
{
#if 0
    std::cerr << "[Error]: SimdF_t pow not implemented\n" ;
    exit(1) ;
#else
    ValueDataF __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper array
    store_a<SimdDataF, ValueDataF>(a, data);
return _mm256_setr_ps(std::pow(data[0], exp), std::pow(data[1], exp),
 std::pow(data[2], exp),
 std::pow(data[3], exp), std::pow(data[4], exp), std::pow(data[5], exp),
 std::pow(data[6], exp), std::pow(data[7], exp));
#endif
}
template <> __KFP_SIMD__INLINE SimdDataF sign<SimdDataF>(const SimdDataF& a)
{
    return ANDBits<SimdDataF>(
        type_cast<SimdDataF, SimdDataI>(getMask<MASK::MINUS>()), a);
}

template <>
__KFP_SIMD__INLINE void print<SimdDataF>(std::ostream& stream, const SimdDataF& val_simd)
{
    ValueDataF __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper array for AVX-256
    store_a<SimdDataF, ValueDataF>(val_simd, data);
    stream << "[" << data[0] << ", " << data[1] << ", " << data[2] << ", "
           << data[3] << ", " << data[4] << ", " << data[5] << ", " << data[6] << ", "
           << data[7] << "]";
}

} // namespace Detail

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_AVX_DETAIL_FLOAT_H
