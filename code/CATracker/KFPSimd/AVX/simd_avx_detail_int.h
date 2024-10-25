// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_AVX_DETAIL_INT_H
#define SIMD_AVX_DETAIL_INT_H

#include "../Base/simd_macros.h"
#include "../Base/simd_detail.h"
#include "simd_avx_type.h"

#include <cmath>
#include <iostream>

namespace KFP {
namespace SIMD {

namespace Detail {

enum class MASK { ABS, MINUS, TRUE, INF };

template <MASK mask>
constexpr __KFP_SIMD__INLINE SimdDataI getMask()
{
  switch (mask) {
  case MASK::ABS:
    return _mm256_set1_epi32(0x7FFFFFFF);
  case MASK::MINUS:
    return _mm256_set1_epi32(0x80000000);
  case MASK::TRUE:
    return _mm256_set1_epi32(-1);
  case MASK::INF:
    return _mm256_set1_epi32(0x7F800000);
  }
}

// ------------------------------------------------------
// General
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI type_cast<SimdDataI, SimdDataF>(const SimdDataF &val_simd) {
  return _mm256_castps_si256(val_simd);
}
template <>
__KFP_SIMD__INLINE SimdDataI type_cast<SimdDataI>(const SimdDataI &val_simd) {
  return val_simd;
}
template <>
__KFP_SIMD__INLINE SimdDataI value_cast<SimdDataI, SimdDataF>(const SimdDataF &val_simd) {
  return _mm256_cvttps_epi32(val_simd);
}
template <>
__KFP_SIMD__INLINE SimdDataI value_cast<SimdDataI>(const SimdDataI &val_simd) {
  return val_simd;
}
template <> __KFP_SIMD__INLINE SimdDataI constant<SimdDataI, ValueDataI>(ValueDataI val) {
  return _mm256_set1_epi32(val);
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI load<SimdDataI, ValueDataI>(const ValueDataI *val_ptr) {
  return _mm256_loadu_si256(reinterpret_cast<const SimdDataI *>(val_ptr));
}
template <>
__KFP_SIMD__INLINE SimdDataI load_a<SimdDataI, ValueDataI>(const ValueDataI *val_ptr) {
  return _mm256_load_si256(reinterpret_cast<const SimdDataI *>(val_ptr));
}
template <>
__KFP_SIMD__INLINE void store<SimdDataI, ValueDataI>(const SimdDataI &val_simd,
                                         ValueDataI *val_ptr) {
  _mm256_storeu_si256(reinterpret_cast<SimdDataI *>(val_ptr), val_simd);
}
template <>
__KFP_SIMD__INLINE void store_a<SimdDataI, ValueDataI>(const SimdDataI &val_simd,
                                           ValueDataI *val_ptr) {
  _mm256_store_si256(reinterpret_cast<SimdDataI *>(val_ptr), val_simd);
}

// ------------------------------------------------------
// Logical bitwise
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI ANDBits<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm256_and_si256(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI ORBits<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm256_or_si256(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI XORBits<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm256_xor_si256(a, b);
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
  return _mm256_cmpeq_epi32(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI notEqual<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return NOTBits<SimdDataI>(equal<SimdDataI>(a, b));
}
template <>
__KFP_SIMD__INLINE SimdDataI lessThan<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm256_cmpgt_epi32(b, a);
}
template <>
__KFP_SIMD__INLINE SimdDataI greaterThan<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm256_cmpgt_epi32(a, b);
}
template <>
__KFP_SIMD__INLINE SimdDataI lessThanEqual<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return NOTBits<SimdDataI>(greaterThan<SimdDataI>(a, b));
}
template <>
__KFP_SIMD__INLINE SimdDataI greaterThanEqual<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return lessThanEqual<SimdDataI>(b, a);
}

// ------------------------------------------------------
// Manipulate bits
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE SimdDataI shiftLBits<SimdDataI>(const SimdDataI &a, int b) {
  return _mm256_slli_epi32(a, b);
}
template <> __KFP_SIMD__INLINE SimdDataI shiftRBits<SimdDataI>(const SimdDataI &a, int b) {
  return _mm256_srai_epi32(a, b);
}

// ------------------------------------------------------
// Logical lanewise
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI ANDLanes<SimdDataI>(const SimdDataI& a, const SimdDataI& b) {
    return ANDBits<SimdDataI>(a, b);
}

template <>
__KFP_SIMD__INLINE SimdDataI ORLanes<SimdDataI>(const SimdDataI& a, const SimdDataI& b) {
    return ORBits<SimdDataI>(a, b);
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
__KFP_SIMD__INLINE SimdDataI select<SimdDataI>(const SimdDataI &mask, const SimdDataI &a, const SimdDataI &b) {
#if defined(__KFP_SIMD__AVX) //TODO
  return _mm256_blendv_epi8(b, a, mask);
#else
  return ORBits<SimdDataI>(ANDBits<SimdDataI>(mask, a), _mm256_andnot_si256(mask, b));
#endif
}
template <int N> __KFP_SIMD__INLINE ValueDataI get(const SimdDataI &a) {
  return _mm256_extract_epi32(a, N);
}
template <>
__KFP_SIMD__INLINE ValueDataI extract<ValueDataI, SimdDataI>(int index, const SimdDataI &val_simd) {
  switch (index) {
  case 0:
    return _mm256_extract_epi32(val_simd, 0);
  case 1:
    return _mm256_extract_epi32(val_simd, 1);
  case 2:
    return _mm256_extract_epi32(val_simd, 2);
  case 3:
    return _mm256_extract_epi32(val_simd, 3);
  case 4:
    return _mm256_extract_epi32(val_simd, 4);
  case 5:
    return _mm256_extract_epi32(val_simd, 5);
  case 6:
    return _mm256_extract_epi32(val_simd, 6);
  case 7:
  default:
    return _mm256_extract_epi32(val_simd, 7);
  }
}
template <>
__KFP_SIMD__INLINE void insert<SimdDataI, ValueDataI>(SimdDataI &val_simd, int index, ValueDataI val) {
#if 1
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) ValueDataI
    indices[__KFP_SIMD__Len_Int]{}; // Helper indices array
    indices[index] = -1;
    const SimdDataI mask = load<SimdDataI, ValueDataI>(indices);
    val_simd = select<SimdDataI>(
        mask, constant<SimdDataI, ValueDataI>(val), val_simd);
#else
    switch (index) {
        case 0:
            val_simd = _mm256_insert_epi32(val_simd, val, 0);
            break;
        case 1:
            val_simd = _mm256_insert_epi32(val_simd, val, 1);
            break;
        case 2:
            val_simd = _mm256_insert_epi32(val_simd, val, 2);
            break;
        case 3:
            val_simd = _mm256_insert_epi32(val_simd, val, 3);
            break;
        case 4:
            val_simd = _mm256_insert_epi32(val_simd, val, 4);
            break;
        case 5:
            val_simd = _mm256_insert_epi32(val_simd, val, 5);
            break;
        case 6:
            val_simd = _mm256_insert_epi32(val_simd, val, 6);
            break;
        case 7:
        default:
            val_simd = _mm256_insert_epi32(val_simd, val, 7);
            break;
    }
#endif
}

// Robin: I'm not sure about these two functions. Additionally, SSE only has one function for shiftLLanes. I put another one below.
// template <>
// __KFP_SIMD__INLINE SimdDataI shiftLLanes<SimdDataI>(int n, const SimdDataI &val_simd) {
//   constexpr int value_size_bytes = sizeof(int);
//   switch (n) {
//   case 0:
//     return val_simd;
//   case 1:
//     return _mm256_permutevar8x32_epi32(val_simd, _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0));
//   case 2:
//     return _mm256_permutevar8x32_epi32(val_simd, _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1));
//   case 3:
//     return _mm256_permutevar8x32_epi32(val_simd, _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2));
//   default:
//     return _mm256_set1_epi32(0);
//   }
// }
// template <>
// __KFP_SIMD__INLINE SimdDataI shiftLLanes<SimdDataI>(int n, const SimdDataI &val_simd) {
//   constexpr int value_size_bytes = sizeof(int) * 8;
//   __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); //TODO Initial indices
//   indices = _mm256_add_epi32(indices, _mm256_set1_epi32(n)); // Add n to each index
//   return _mm256_permutevar8x32_epi32(val_simd, indices); // Permute based on indices
// }

// Robin:
template <>
__KFP_SIMD__INLINE __m256i shiftLLanes<__m256i>(size_t n, const __m256i &val_simd) {
  constexpr size_t value_size_bytes = sizeof(int);
  constexpr size_t lane_count = 8; // AVX has 8 lanes for 32-bit integers
//  constexpr size_t total_size_bytes = lane_count * value_size_bytes;

  if (n >= lane_count) {
    return _mm256_setzero_si256();
  }

  switch (n) {
  case 0:
    return val_simd;
  case 1:
    return _mm256_srli_si256(val_simd, value_size_bytes);
  case 2:
    return _mm256_srli_si256(val_simd, 2 * value_size_bytes);
  case 3:
    return _mm256_srli_si256(val_simd, 3 * value_size_bytes);
  case 4:
    return _mm256_srli_si256(val_simd, 4 * value_size_bytes);
  case 5:
    return _mm256_srli_si256(val_simd, 5 * value_size_bytes);
  case 6:
    return _mm256_srli_si256(val_simd, 6 * value_size_bytes);
  case 7:
    return _mm256_srli_si256(val_simd, 7 * value_size_bytes);
  default:
    return _mm256_setzero_si256();
  }
}

// Robin: I also don't know about this one. I put another one below.
// template <> __KFP_SIMD__INLINE SimdDataI rotate<SimdDataI>(int n, const SimdDataI &val_simd) {
// #if defined(__KFP_SIMD__AVX)
//   constexpr int value_size_bytes = sizeof(int);
//   __m256i tmp1, tmp2;
//   switch (n) {
//   case 0:
//     return val_simd;
//   case 1:
//     tmp1 = _mm256_permute2x128_si256(val_simd, val_simd, 0x03);
//     tmp2 = _mm256_alignr_epi8(val_simd, tmp1, value_size_bytes);
//     return tmp2;
//   case 2:
//     return _mm256_permute4x64_epi64(val_simd, _MM_SHUFFLE(2, 1, 0, 3));
//   case 3:
//     tmp1 = _mm256_permute2x128_si256(val_simd, val_simd, 0x03);
//     tmp2 = _mm256_alignr_epi8(val_simd, tmp1, 3 * value_size_bytes);
//     return tmp2;
//   }
//   return _mm256_setzero_si256();
// #else
//   switch (n) {
//   case 0:
//     return val_simd;
//   case 1:
//     return ORBits<SimdDataI>(shiftLLanes<SimdDataI>(1, val_simd),
//                              shiftRLanes<SimdDataI>(3, val_simd));
//   case 2:
//     return ORBits<SimdDataI>(shiftLLanes<SimdDataI>(2, val_simd),
//                              shiftRLanes<SimdDataI>(2, val_simd));
//   case 3:
//     return ORBits<SimdDataI>(shiftLLanes<SimdDataI>(3, val_simd),
//                              shiftRLanes<SimdDataI>(1, val_simd));
//   }
//   return constant<SimdDataI, ValueDataI>(0);
// #endif
// }

// Robin:
template <>
__KFP_SIMD__INLINE __m256i rotate<__m256i>(size_t n, const __m256i &val_simd) {
//  constexpr size_t value_size_bytes = sizeof(int);
  constexpr size_t lane_count = 8;

  if (n >= lane_count) {
    return _mm256_setzero_si256();
  }

  switch (n) {
  case 0:
    return val_simd;
  case 1:
    return _mm256_or_si256(_mm256_srli_si256(val_simd, 4), _mm256_slli_si256(val_simd, 28));
  case 2:
    return _mm256_or_si256(_mm256_srli_si256(val_simd, 8), _mm256_slli_si256(val_simd, 24));
  case 3:
    return _mm256_or_si256(_mm256_srli_si256(val_simd, 12), _mm256_slli_si256(val_simd, 20));
  case 4:
    return _mm256_or_si256(_mm256_srli_si256(val_simd, 16), _mm256_slli_si256(val_simd, 16));
  case 5:
    return _mm256_or_si256(_mm256_srli_si256(val_simd, 20), _mm256_slli_si256(val_simd, 12));
  case 6:
    return _mm256_or_si256(_mm256_srli_si256(val_simd, 24), _mm256_slli_si256(val_simd, 8));
  case 7:
    return _mm256_or_si256(_mm256_srli_si256(val_simd, 28), _mm256_slli_si256(val_simd, 4));
  default:
    return _mm256_setzero_si256();
  }
}

// ------------------------------------------------------
// Basic Arithmetic
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE SimdDataI add<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm256_add_epi32(a, b);
}

template <>
__KFP_SIMD__INLINE SimdDataI substract<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  return _mm256_sub_epi32(a, b);
}

template <>
__KFP_SIMD__INLINE SimdDataI multiply<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
#if defined(__KFP_SIMD__AVX) // TODO
  return _mm256_mullo_epi32(a, b);
#else
  ValueDataI __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
      data1[__KFP_SIMD__Len_Int]{}; // Helper array
  ValueDataI __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
      data2[__KFP_SIMD__Len_Int]{}; // Helper array
  store_a<SimdDataI, ValueDataI>(a, data1);
  store_a<SimdDataI, ValueDataI>(b, data2);
  const ValueDataI __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
      data[__KFP_SIMD__Len_Int]{data1[0] * data2[0], data1[1] * data2[1],
                                data1[2] * data2[2], data1[3] * data2[3],
                                data1[4] * data2[4], data1[5] * data2[5],
                                data1[6] * data2[6], data1[7] * data2[7]};
  return load_a<SimdDataI, ValueDataI>(data);
#endif
}
template <>
__KFP_SIMD__INLINE SimdDataI divide<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
  ValueDataI __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
      data1[__KFP_SIMD__Len_Int]{}; // Helper array
  ValueDataI __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
      data2[__KFP_SIMD__Len_Int]{}; // Helper array
  store_a<SimdDataI, ValueDataI>(a, data1);
  store_a<SimdDataI, ValueDataI>(b, data2);
  const ValueDataI __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
      data[__KFP_SIMD__Len_Int]{data1[0] / data2[0], data1[1] / data2[1],
                                data1[2] / data2[2], data1[3] / data2[3],
                                data1[4] / data2[4], data1[5] / data2[5],
                                data1[6] / data2[6], data1[7] / data2[7]};
  return load_a<SimdDataI, ValueDataI>(data);
}
template <> __KFP_SIMD__INLINE SimdDataI minus<SimdDataI>(const SimdDataI &a) {
  return substract<SimdDataI>(_mm256_setzero_si256(), a);
}

template <>
__KFP_SIMD__INLINE SimdDataI min<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
#if defined(__KFP_SIMD__AVX) //TODO
  return _mm256_min_epi32(a, b);
#else
  const SimdDataI mask = greaterThan<SimdDataI>(a, b);
  return select<SimdDataI>(mask, b, a);
#endif
}
template <>
__KFP_SIMD__INLINE SimdDataI max<SimdDataI>(const SimdDataI &a, const SimdDataI &b) {
#if defined(__KFP_SIMD__AVX)
  return _mm256_max_epi32(a, b);
#else
  const SimdDataI mask = greaterThan<SimdDataI>(a, b);
  return select<SimdDataI>(mask, a, b);
#endif
}

///
template<>
inline ValueDataI min<ValueDataI, SimdDataI>(const SimdDataI& a)
{
    __m256i shuf1 = _mm256_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
    __m256i min1 = _mm256_min_epi32(a, shuf1);
    __m256i shuf2 = _mm256_shuffle_epi32(min1, _MM_SHUFFLE(2, 3, 0, 1));
    __m256i min2 = _mm256_min_epi32(min1, shuf2);
    __m128i min_low = _mm256_castsi256_si128(min2);
    __m128i min_high = _mm256_extracti128_si256(min2, 1);
    __m128i final_min = _mm_min_epi32(min_low, min_high);
    __m128i shuf3 = _mm_shuffle_epi32(final_min, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i min_final = _mm_min_epi32(final_min, shuf3);
    return _mm_cvtsi128_si32(min_final);
}
template<>
inline ValueDataI max<ValueDataI, SimdDataI>(const SimdDataI& a)
{
    __m256i shuf1 = _mm256_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
    __m256i max1 = _mm256_max_epi32(a, shuf1);
    __m256i shuf2 = _mm256_shuffle_epi32(max1, _MM_SHUFFLE(2, 3, 0, 1));
    __m256i max2 = _mm256_max_epi32(max1, shuf2);
    __m128i max_low = _mm256_castsi256_si128(max2);
    __m128i max_high = _mm256_extracti128_si256(max2, 1);
    __m128i final_max = _mm_max_epi32(max_low, max_high);
    __m128i shuf3 = _mm_shuffle_epi32(final_max, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i max_final = _mm_max_epi32(final_max, shuf3);
    return _mm_cvtsi128_si32(max_final);
}
///

template <> __KFP_SIMD__INLINE SimdDataI sqrt<SimdDataI>(const SimdDataI &a) {
  // Note: There's no direct AVX2 equivalent for integer sqrt, so we use the same approach as SSE
  return _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(a)));
}

template <> __KFP_SIMD__INLINE SimdDataI rsqrt<SimdDataI>(const SimdDataI &a) {
  // Note: There's no direct AVX2 equivalent for integer rsqrt, so we use the same approach as SSE
  return _mm256_cvtps_epi32(_mm256_rsqrt_ps(_mm256_cvtepi32_ps(a)));
}

template <> __KFP_SIMD__INLINE SimdDataI abs<SimdDataI>(const SimdDataI &a) {
#if defined(__KFP_SIMD__AVX2)
  return _mm256_abs_epi32(a);
#else
  return ANDBits<SimdDataI>(a, getMask<MASK::ABS>());
#endif
}
template <> __KFP_SIMD__INLINE SimdDataI log<SimdDataI>(const SimdDataI &a) {
  ValueDataI __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
      data[__KFP_SIMD__Len_Int]{}; // Helper array //TODO
  store_a<SimdDataI, ValueDataI>(a, data);
  // Note: Direct AVX2 log operation on integers is not available, using SSE approach
  return _mm256_setr_epi32(std::log(data[0]), std::log(data[1]), std::log(data[2]),
                           std::log(data[3]), std::log(data[4]), std::log(data[5]),
                           std::log(data[6]), std::log(data[7]));
}
template <> __KFP_SIMD__INLINE SimdDataI pow<SimdDataI>(const SimdDataI &a, int exp) {
  ValueDataI __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
      data[__KFP_SIMD__Len_Int]{}; // Helper array
  store_a<SimdDataI, ValueDataI>(a, data);
  // Note: Direct AVX2 pow operation on integers is not available, using SSE approach
  return _mm256_setr_epi32(std::pow(data[0], exp), std::pow(data[1], exp),
                           std::pow(data[2], exp), std::pow(data[3], exp),
                           std::pow(data[4], exp), std::pow(data[5], exp),
                           std::pow(data[6], exp), std::pow(data[7], exp));
}
template <> __KFP_SIMD__INLINE int sign<int, SimdDataI>(const SimdDataI &a) {
  // Note: _mm256_movemask_ps is not directly available, using _mm_movemask_ps on parts
  return (_mm_movemask_ps(_mm_castsi128_ps(_mm256_extractf128_si256(a, 0))) |
          (_mm_movemask_ps(_mm_castsi128_ps(_mm256_extractf128_si256(a, 1))) << 4));
}
template <> __KFP_SIMD__INLINE SimdDataI sign<SimdDataI>(const SimdDataI &a) {
  // Note: Using AVX2 operation for sign, as direct AVX equivalent is not available
  return ANDBits<SimdDataI>(getMask<MASK::MINUS>(), a);
}

template <>
__KFP_SIMD__INLINE bool equal<bool, SimdDataI, SimdDataI>(const SimdDataI& a, const SimdDataI& b) {
    return (_mm256_movemask_ps(_mm256_castsi256_ps(a)) == _mm256_movemask_ps(_mm256_castsi256_ps(b)));
}

template <>
__KFP_SIMD__INLINE bool notEqual<bool, SimdDataI, SimdDataI>(const SimdDataI& a, const SimdDataI& b) {
    return (_mm256_movemask_ps(_mm256_castsi256_ps(a)) != _mm256_movemask_ps(_mm256_castsi256_ps(b)));
}

template <>
__KFP_SIMD__INLINE void print<SimdDataI>(std::ostream &stream, const SimdDataI &val_simd) {
    ValueDataI __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{};
    store_a<SimdDataI, ValueDataI>(val_simd, data);
    stream << "[" << data[0] << ", " << data[1] << ", " << data[2] << ", "
           << data[3] << ", " << data[4] << ", " << data[5] << ", "
           << data[6] << ", " << data[7] << "]";
}

} // namespace Detail

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_SSE_DETAIL_INT_H
