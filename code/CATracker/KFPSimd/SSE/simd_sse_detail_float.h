// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SSE_DETAIL_FLOAT_H
#define SIMD_SSE_DETAIL_FLOAT_H

#include "../Base/simd_macros.h"
#include "../Base/simd_detail.h"
#include "simd_sse_detail_int.h"
#include "simd_sse_type.h"

#include <x86intrin.h>
#include <cmath>
#include <iostream>

namespace KFP {
namespace SIMD {

namespace Detail {

// ------------------------------------------------------
// General
// ------------------------------------------------------
template <> inline SimdDataF type_cast<SimdDataF, SimdDataI>(const SimdDataI& val_simd)
{
    return _mm_castsi128_ps(val_simd);
}
template <>
inline SimdDataF type_cast<SimdDataF>(const SimdDataF &val_simd) {
  return val_simd;
}
template <> inline SimdDataF value_cast<SimdDataF, SimdDataI>(const SimdDataI& val_simd)
{
    return _mm_cvtepi32_ps(val_simd);
}
template <>
inline SimdDataF value_cast<SimdDataF>(const SimdDataF &val_simd) {
  return val_simd;
}
template <> inline SimdDataF constant<SimdDataF, ValueDataF>(ValueDataF val)
{
    return _mm_set1_ps(val);
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
template <>
inline SimdDataF load<SimdDataF, ValueDataF>(const ValueDataF* val_ptr)
{
    return _mm_loadu_ps(val_ptr);
}
template <>
inline SimdDataF load_a<SimdDataF, ValueDataF>(const ValueDataF* val_ptr)
{
    return _mm_load_ps(val_ptr);
}
template <>
inline void store<SimdDataF, ValueDataF>(const SimdDataF& val_simd,
                                         ValueDataF* val_ptr)
{
    _mm_storeu_ps(val_ptr, val_simd);
}
template <>
inline void store_a<SimdDataF, ValueDataF>(const SimdDataF& val_simd,
                                           ValueDataF* val_ptr)
{
    _mm_store_ps(val_ptr, val_simd);
}

// ------------------------------------------------------
// Logical bitwise
// ------------------------------------------------------
template <>
inline SimdDataF ANDBits<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_and_ps(a, b);
}
template <>
inline SimdDataF ORBits<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_or_ps(a, b);
}
template <>
inline SimdDataF XORBits<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_xor_ps(a, b);
}
// template <>
// inline SimdDataF NOTBits<SimdDataF>(const SimdDataF& a)
// {
    // return XORBits<SimdDataF>(type_cast<SimdDataF, SimdDataI>(getMask<MASK::TRUE>()), a);
// }

// ------------------------------------------------------
// Comparison
// ------------------------------------------------------
template <>
inline SimdDataF equal<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_cmpeq_ps(a, b);
}
template <>
inline SimdDataF notEqual<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_cmpneq_ps(a, b);
}
template <>
inline SimdDataF lessThan<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_cmplt_ps(a, b);
}
template <>
inline SimdDataF lessThanEqual<SimdDataF>(const SimdDataF& a,
                                            const SimdDataF& b)
{
    return _mm_cmple_ps(a, b);
}
template <>
inline SimdDataF greaterThan<SimdDataF>(const SimdDataF& a,
                                          const SimdDataF& b)
{
    return _mm_cmpgt_ps(a, b);
}
template <>
inline SimdDataF greaterThanEqual<SimdDataF>(const SimdDataF& a,
                                               const SimdDataF& b)
{
    return _mm_cmpge_ps(a, b);
}

// ------------------------------------------------------
// Manipulate lanes
// ------------------------------------------------------
template <>
inline SimdDataF select(const SimdDataF& mask, const SimdDataF& a,
                        const SimdDataF& b)
{
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
    return _mm_blendv_ps(b, a, mask);
#else
    return ORBits<SimdDataF>(ANDBits<SimdDataF>(mask, a), _mm_andnot_ps(mask, b));
#endif
}
template <int N> inline ValueDataF get(const SimdDataF& a)
{
    const SimdDataF result = _mm_shuffle_ps(a, a, (N & 3));
    return _mm_cvtss_f32(result);
}
template <>
inline ValueDataF extract<ValueDataF, SimdDataF>(int index, const SimdDataF& a)
{
#if 0
    ValueDataF __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
    data[__KFP_SIMD__Len_Float]{}; // Helper array
    store_a(data);
    return data[index];
// #elif defined(__KFP_SIMD__SSE4_1)
#elif 0
    float result;
    switch (index) {
    case 0:
        _MM_EXTRACT_FLOAT(result, a, 0x00);
        break;
    case 1:
        _MM_EXTRACT_FLOAT(result, a, 0x01);
        break;
    case 2:
        _MM_EXTRACT_FLOAT(result, a, 0x02);
        break;
    case 3:
    default:
        _MM_EXTRACT_FLOAT(result, a, 0x03);
        break;
    }
    return result;
#else
    switch (index) {
    case 0:
        return get<0>(a);
    case 1:
        return get<1>(a);
    case 2:
        return get<2>(a);
    case 3:
    default:
        return get<3>(a);
    }
#endif
}
template <>
inline void insert<SimdDataF, ValueDataF>(SimdDataF &val_simd, int index,
                                          const ValueDataF &val) {
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
    switch (index) {
    case 0:
        val_simd = _mm_insert_ps(val_simd, _mm_set_ss(val), 0 << 4);
        break;
    case 1:
        val_simd = _mm_insert_ps(val_simd, _mm_set_ss(val), 1 << 4);
        break;
    case 2:
        val_simd = _mm_insert_ps(val_simd, _mm_set_ss(val), 2 << 4);
        break;
    case 3:
    default:
        val_simd = _mm_insert_ps(val_simd, _mm_set_ss(val), 3 << 4);
        break;
    }
#else
  int __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
      indices[__KFP_SIMD__Len_Int] = {0, 0, 0, 0};
  indices[index] = -1;
  const SimdDataF mask = type_cast<SimdDataF, SimdDataI>(load_a<SimdDataI, ValueDataI>(indices));
  val_simd = select<SimdDataF>(
      mask, constant<SimdDataF, ValueDataF>(val), val_simd);
#endif
}
template <>
inline SimdDataF shiftLLanes<SimdDataF>(int amount, const SimdDataF &val_simd) {
    SimdDataI tmp = type_cast<SimdDataI, SimdDataF>(val_simd);
    return type_cast<SimdDataF, SimdDataI>(shiftLLanes<SimdDataI>(amount, tmp));
}
template <>
inline SimdDataF shiftRLanes<SimdDataF>(int amount, const SimdDataF &val_simd) {
    SimdDataI tmp = type_cast<SimdDataI, SimdDataF>(val_simd);
    return type_cast<SimdDataF, SimdDataI>(shiftRLanes<SimdDataI>(amount, tmp));
}
template <> inline SimdDataF rotate<SimdDataF>(int amount, const SimdDataF& val_simd)
{
    SimdDataI tmp = type_cast<SimdDataI, SimdDataF>(val_simd);
    return type_cast<SimdDataF, SimdDataI>(rotate<SimdDataI>(amount, tmp));
}

// ------------------------------------------------------
// Basic Arithmetic
// ------------------------------------------------------
template <>
inline SimdDataF add<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_add_ps(a, b);
}
template <>
inline SimdDataF substract<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_sub_ps(a, b);
}
template <>
inline SimdDataF multiply<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_mul_ps(a, b);
}
template <>
inline SimdDataF divide<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_div_ps(a, b);
}
template <> inline SimdDataF minus<SimdDataF>(const SimdDataF& a)
{
    return XORBits<SimdDataF>(a, type_cast<SimdDataF, SimdDataI>(getMask<MASK::MINUS>()));
}

template <>
inline SimdDataF min<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_min_ps(a, b);
}
template <>
inline SimdDataF max<SimdDataF>(const SimdDataF& a, const SimdDataF& b)
{
    return _mm_max_ps(a, b);
}

template <> inline SimdDataF sqrt<SimdDataF>(const SimdDataF& a)
{
    return _mm_sqrt_ps(a);
}
template <> inline SimdDataF rsqrt<SimdDataF>(const SimdDataF& a)
{
    return _mm_rsqrt_ps(a);
}
template <> inline SimdDataF abs<SimdDataF>(const SimdDataF& a)
{
    return ANDBits<SimdDataF>(a, type_cast<SimdDataF, SimdDataI>(getMask<MASK::ABS>()));
}
template <> inline SimdDataF log<SimdDataF>(const SimdDataF& a)
{
    ValueDataF __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper array
    store_a<SimdDataF, ValueDataF>(a, data);
    return _mm_setr_ps(std::log(data[0]), std::log(data[1]), std::log(data[2]),
                       std::log(data[3]));
}
template <> inline SimdDataF pow<SimdDataF>(const SimdDataF& a, int exp)
{
#if 0
    std::cerr << "[Error]: SimdF_t pow not implemented\n" ;
    exit(1) ;
#else
    ValueDataF __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper array
    store_a<SimdDataF, ValueDataF>(a, data);
    return _mm_setr_ps(std::pow(data[0], exp), std::pow(data[1], exp),
                       std::pow(data[2], exp), std::pow(data[3], exp));
#endif
}
template <> inline SimdDataF sign<SimdDataF>(const SimdDataF& a)
{
    return ANDBits<SimdDataF>(
        type_cast<SimdDataF, SimdDataI>(getMask<MASK::MINUS>()), a);
}

template <>
inline void print<SimdDataF>(std::ostream& stream, const SimdDataF& val_simd)
{
    ValueDataF __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper array
    store_a<SimdDataF, ValueDataF>(val_simd, data);
    stream << "[" << data[0] << ", " << data[1] << ", " << data[2] << ", "
           << data[3] << "]";
}

} // namespace Detail

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_SSE_DETAIL_FLOAT_H
