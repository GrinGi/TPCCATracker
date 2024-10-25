// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran, P. Kisel, I. Kisel
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SSE_DETAIL_INT_H
#define SIMD_SSE_DETAIL_INT_H

#include "../Base/simd_macros.h"
#include "../Base/simd_detail.h"
#include "simd_sse_type.h"

#include <cmath>
#include <iostream>

namespace KFP {
namespace SIMD {

namespace Detail {

enum class MASK { ABS, MINUS, TRUE, INF };

template <MASK mask>
const __KFP_SIMD__INLINE SimdDataI getMask()
{
  switch (mask) {
  case MASK::ABS:
//     {
//         __m128i minus1;
// #if _MSC_VER && !__INTEL_COMPILER
//         minus1 = _mm_setzero_si128();
// #endif
//         minus1 = _mm_cmpeq_epi32(minus1, minus1);
//         return (_mm_srli_epi32(minus1, 1));
//     }
    return _mm_set1_epi32(0x7FFFFFFF);
  case MASK::MINUS:
    return _mm_set1_epi32(0x80000000);
  case MASK::TRUE:
    return _mm_set1_epi32(-1);
  case MASK::INF:
    return _mm_set1_epi32(0x7F800000);
  }
}

// ------------------------------------------------------
// General
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI type_cast<SimdDataI, SimdDataF>(const SimdDataF &val_simd) {
  return _mm_castps_si128(val_simd);
}
template <>
__KFP_SIMD__INLINE SimdDataI type_cast<SimdDataI, SimdDataI>(const SimdDataI &val_simd) {
  return val_simd;
}
template <>
__KFP_SIMD__INLINE SimdDataI value_cast<SimdDataI, SimdDataF>(const SimdDataF &val_simd) {
  return _mm_cvttps_epi32(val_simd);
}
template <>
__KFP_SIMD__INLINE SimdDataI value_cast<SimdDataI, SimdDataI>(const SimdDataI &val_simd) {
  return val_simd;
}
template <> __KFP_SIMD__INLINE SimdDataI constant<SimdDataI, ValueDataI>(ValueDataI val) {
  return _mm_set1_epi32(val);
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI load<SimdDataI, ValueDataI>(const ValueDataI *val_ptr) {
  return _mm_loadu_si128(reinterpret_cast<const SimdDataI *>(val_ptr));
}
template <>
__KFP_SIMD__INLINE SimdDataI load_a<SimdDataI, ValueDataI>(const ValueDataI *val_ptr) {
  return _mm_load_si128(reinterpret_cast<const SimdDataI *>(val_ptr));
}
template <>
__KFP_SIMD__INLINE void store<SimdDataI, ValueDataI>(const SimdDataI &val_simd,
                                         ValueDataI *val_ptr) {
  _mm_storeu_si128(reinterpret_cast<SimdDataI *>(val_ptr), val_simd);
}
template <>
__KFP_SIMD__INLINE void store_a<SimdDataI, ValueDataI>(const SimdDataI &val_simd,
                                           ValueDataI *val_ptr) {
  _mm_store_si128(reinterpret_cast<SimdDataI *>(val_ptr), val_simd);
}

// ------------------------------------------------------
// Logical bitwise
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI ANDBits<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm_and_si128(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI ORBits<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm_or_si128(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI XORBits<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm_xor_si128(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI NOTBits<SimdDataI>(const SimdDataI& a)
{
    return XORBits<SimdDataI>(getMask<MASK::TRUE>(), a);
}

// ------------------------------------------------------
// Comparison
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI equal<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm_cmpeq_epi32(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI notEqual<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return NOTBits<SimdDataI>(equal<SimdDataI>(a, b));
}
template <>
__KFP_SIMD__INLINE SimdDataI lessThan<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm_cmplt_epi32(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI greaterThan<SimdDataI>(const SimdDataI &a,
                                        const SimdDataI &b) {
  return _mm_cmpgt_epi32(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI lessThanEqual<SimdDataI>(const SimdDataI &a,
                                          const SimdDataI &b) {
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
  return equal<SimdDataI>(_mm_min_epi32(a, b), a);
#else
  return NOTBits<SimdDataI>(greaterThan<SimdDataI>(a, b));
#endif
}
template <>
__KFP_SIMD__INLINE SimdDataI greaterThanEqual<SimdDataI>(const SimdDataI &a,
                                             const SimdDataI &b) {
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
  return equal<SimdDataI>(_mm_min_epi32(b, a), b);
#else
  return lessThanEqual<SimdDataI>(b, a);
#endif
}

// ------------------------------------------------------
// Manipulate bits
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE SimdDataI shiftLBits<SimdDataI>(const SimdDataI &a, int b) {
  return _mm_slli_epi32(a, b);
}
template <> __KFP_SIMD__INLINE SimdDataI shiftRBits<SimdDataI>(const SimdDataI &a, int b) {
  return _mm_srai_epi32(a, b);
}

// ------------------------------------------------------
// Logical lanewise
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI ANDLanes<SimdDataI>(const SimdDataI& a,
                                                 const SimdDataI& b)
{
    return ANDBits<SimdDataI>(a,b);
}
template <>
__KFP_SIMD__INLINE SimdDataI ORLanes<SimdDataI>(const SimdDataI& a,
                                                 const SimdDataI& b)
{
    return ORBits<SimdDataI>(a,b);
}
template <>
__KFP_SIMD__INLINE SimdDataI XORLanes<SimdDataI>(const SimdDataI& a,
                                                 const SimdDataI& b)
{
    return XORBits<SimdDataI>(a,b);
}
template <>
__KFP_SIMD__INLINE SimdDataI NOTLanes<SimdDataI>(const SimdDataI& a)
{
    return NOTBits<SimdDataI>(a);
}

// ------------------------------------------------------
// Manipulate lanes
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI select<SimdDataI>(const SimdDataI &mask, const SimdDataI &a,
                                   const SimdDataI &b) {
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
  return _mm_blendv_epi8(b, a, mask);
#else
  return ORBits<SimdDataI>(ANDBits<SimdDataI>(mask, a), _mm_andnot_si128(mask, b));
#endif
}
template <int N> __KFP_SIMD__INLINE ValueDataI get(const SimdDataI &a) {
  const SimdDataI result = _mm_shuffle_epi32(a, N);
  return _mm_cvtsi128_si32(result);
}
template <>
__KFP_SIMD__INLINE ValueDataI extract<ValueDataI, SimdDataI>(int index,
                                                 const SimdDataI &val_simd) {
#if 1
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data[__KFP_SIMD__Len_Int]{}; // Helper array
    store_a<SimdDataI, ValueDataI>(val_simd, data);
    return data[index];
#elif defined(__KFP_SIMD__SSE4_1)
  switch (index) {
  case 0:
    return _mm_extract_epi32(val_simd, 0x00);
  case 1:
    return _mm_extract_epi32(val_simd, 0x01);
  case 2:
    return _mm_extract_epi32(val_simd, 0x02);
  case 3:
  default:
    return _mm_extract_epi32(val_simd, 0x03);
  }
#else
  switch (index) {
  case 0:
    return get<0>(val_simd);
  case 1:
    return get<1>(val_simd);
  case 2:
    return get<2>(val_simd);
  case 3:
  default:
    return get<3>(val_simd);
  }
#endif
}
template <>
__KFP_SIMD__INLINE void insert<SimdDataI, ValueDataI>(SimdDataI& val_simd, int index,
                                          ValueDataI val) {
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a<SimdDataI, ValueDataI>(val_simd, data);
    data[index] = val;
    val_simd = load_a<SimdDataI, ValueDataI>(data);
}
template <>
__KFP_SIMD__INLINE SimdDataI shiftLLanes<SimdDataI>(int n, const SimdDataI &val_simd) {
  constexpr int value_size_bytes = sizeof(int);
  switch (n) {
  case 0:
    return val_simd;
  case 1:
    return _mm_bsrli_si128(val_simd, value_size_bytes);
  case 2:
    return _mm_bsrli_si128(val_simd, 2 * value_size_bytes);
  case 3:
    return _mm_bsrli_si128(val_simd, 3 * value_size_bytes);
  default:
    return constant<SimdDataI, ValueDataI>(0);
  }
}
template <>
__KFP_SIMD__INLINE SimdDataI shiftRLanes<SimdDataI>(int n, const SimdDataI &val_simd) {
  constexpr int value_size_bytes = sizeof(int);
  switch (n) {
  case 0:
    return val_simd;
  case 1:
    return _mm_bslli_si128(val_simd, value_size_bytes);
  case 2:
    return _mm_bslli_si128(val_simd, 2 * value_size_bytes);
  case 3:
    return _mm_bslli_si128(val_simd, 3 * value_size_bytes);
  default:
    return constant<SimdDataI, ValueDataI>(0);
  }
}
template <> __KFP_SIMD__INLINE SimdDataI rotate<SimdDataI>(int n, const SimdDataI &val_simd) {
#if defined(__KFP_SIMD__SSSE3) // SSSE3
  constexpr int value_size_bytes = sizeof(int);
  switch (n) {
  case 0:
    return val_simd;
  case 1:
    return _mm_alignr_epi8(val_simd, val_simd, value_size_bytes);
  case 2:
    return _mm_alignr_epi8(val_simd, val_simd, 2 * value_size_bytes);
  case 3:
    return _mm_alignr_epi8(val_simd, val_simd, 3 * value_size_bytes);
  }
    return constant<SimdDataI, ValueDataI>(0);
#else
  switch (n) {
  case 0:
    return val_simd;
  case 1:
    return ORBits<SimdDataI>(shiftLLanes<SimdDataI>(1, val_simd),
                             shiftRLanes<SimdDataI>(3, val_simd));
  case 2:
    return ORBits<SimdDataI>(shiftLLanes<SimdDataI>(2, val_simd),
                             shiftRLanes<SimdDataI>(2, val_simd));
  case 3:
    return ORBits<SimdDataI>(shiftLLanes<SimdDataI>(3, val_simd),
                             shiftRLanes<SimdDataI>(1, val_simd));
  }
    return constant<SimdDataI, ValueDataI>(0);
#endif
}

// ------------------------------------------------------
// Basic Arithmetic
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI add<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm_add_epi32(a, b);
}

template <>
__KFP_SIMD__INLINE SimdDataI substract<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm_sub_epi32(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI multiply<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
  return _mm_mullo_epi32(a, b);
#else

    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data1[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a<SimdDataI, ValueDataI>(a, data1);

    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data2[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a<SimdDataI, ValueDataI>(b, data2);

    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) const ValueDataI
    data[__KFP_SIMD__Len_Int]{data1[0] * data2[0], data1[1] * data2[1],
                                data1[2] * data2[2], data1[3] * data2[3]};
  return load_a<SimdDataI, ValueDataI>(data);
#endif
}
template <>
__KFP_SIMD__INLINE SimdDataI divide<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data1[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a<SimdDataI, ValueDataI>(a, data1);
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data2[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a<SimdDataI, ValueDataI>(b, data2);
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) const ValueDataI
    data[__KFP_SIMD__Len_Int]{data1[0] / data2[0], data1[1] / data2[1],
                            data1[2] / data2[2], data1[3] / data2[3]};
    return load_a<SimdDataI, ValueDataI>(data);
}
template <> __KFP_SIMD__INLINE SimdDataI minus<SimdDataI>(const SimdDataI &a) {
  return substract<SimdDataI>(_mm_setzero_si128(), a);
}

template <>
__KFP_SIMD__INLINE SimdDataI min<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
  return _mm_min_epi32(a, b);
#else
  const SimdDataI mask = greaterThan<SimdDataI>(a, b);
  return select<SimdDataI>(mask, b, a);
#endif
}
template <>
__KFP_SIMD__INLINE SimdDataI max<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
  return _mm_max_epi32(a, b);
#else
  const SimdDataI mask = greaterThan<SimdDataI>(a, b);
  return select<SimdDataI>(mask, a, b);
#endif
}

///
template<>
inline ValueDataI min<ValueDataI, SimdDataI>(const SimdDataI& a)
{
  __m128i shuf1 = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
  __m128i min1 = _mm_min_epi32(a, shuf1);
  __m128i shuf2 = _mm_shuffle_epi32(min1, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i min2 = _mm_min_epi32(min1, shuf2);
  return _mm_cvtsi128_si32(min2);
}
template<>
inline ValueDataI max<ValueDataI, SimdDataI>(const SimdDataI& a)
{
  __m128i shuf1 = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
  __m128i max1 = _mm_max_epi32(a, shuf1);
  __m128i shuf2 = _mm_shuffle_epi32(max1, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i max2 = _mm_max_epi32(max1, shuf2);
  return _mm_cvtsi128_si32(max2);
}
///

template <> __KFP_SIMD__INLINE SimdDataI sqrt<SimdDataI>(const SimdDataI &a) {
#if 0
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a<SimdDataI, ValueDataI>(a, data);
    return _mm_setr_epi32(std::sqrt(data[0]), std::sqrt(data[1]),
                          std::sqrt(data[2]), std::sqrt(data[3]));
#endif
  return _mm_cvtps_epi32(_mm_sqrt_ps(_mm_cvtepi32_ps(a)));
}
template <> __KFP_SIMD__INLINE SimdDataI rsqrt<SimdDataI>(const SimdDataI &a) {
  return _mm_cvtps_epi32(_mm_rsqrt_ps(_mm_cvtepi32_ps(a)));
}
template <> __KFP_SIMD__INLINE SimdDataI abs<SimdDataI>(const SimdDataI &a) {
#if defined(__KFP_SIMD__SSSE3) // SSSE3
  return _mm_abs_epi32(a);
#else
  return ANDBits<SimdDataI>(a, getMask<MASK::ABS>());
#endif
}
template <> __KFP_SIMD__INLINE SimdDataI log<SimdDataI>(const SimdDataI &a) {
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a<SimdDataI, ValueDataI>(a, data);
    return _mm_setr_epi32(std::log(data[0]), std::log(data[1]), std::log(data[2]),
                        std::log(data[3]));
}
template <> __KFP_SIMD__INLINE SimdDataI pow<SimdDataI>(const SimdDataI &a, int exp) {
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a<SimdDataI, ValueDataI>(a, data);
    return _mm_setr_epi32(std::pow(data[0], exp), std::pow(data[1], exp),
                        std::pow(data[2], exp), std::pow(data[3], exp));
}
template <> __KFP_SIMD__INLINE ValueDataI sign<int, SimdDataI>(const SimdDataI &a) {
  return _mm_movemask_ps(_mm_castsi128_ps(a));
}
template <> __KFP_SIMD__INLINE SimdDataI sign<SimdDataI>(const SimdDataI &a) {
  return ANDBits<SimdDataI>(getMask<MASK::MINUS>(), a);
}

template <>
__KFP_SIMD__INLINE bool equal<bool, SimdDataI, SimdDataI>(const SimdDataI& a,
                                              const SimdDataI& b)
{
    return ( sign<ValueDataI, SimdDataI>(a) == sign<ValueDataI, SimdDataI>(b) );
}
template <>
__KFP_SIMD__INLINE bool notEqual<bool, SimdDataI, SimdDataI>(const SimdDataI& a,
                                                 const SimdDataI& b)
{
    return ( sign<ValueDataI, SimdDataI>(a) != sign<ValueDataI, SimdDataI>(b) );
}

template <>
__KFP_SIMD__INLINE void print<SimdDataI>(std::ostream &stream, const SimdDataI &val_simd) {
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a<SimdDataI, ValueDataI>(val_simd, data);
    stream << "[" << data[0] << ", " << data[1] << ", " << data[2] << ", "
         << data[3] << "]";
}

} // namespace Detail

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_SSE_DETAIL_INT_H
