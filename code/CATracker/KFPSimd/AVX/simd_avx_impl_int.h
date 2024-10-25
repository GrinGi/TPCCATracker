// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_AVX_IMPL_INT_H
#define SIMD_AVX_IMPL_INT_H

#include "../Base/simd_class.h"
#include "simd_avx_type.h"
#include "simd_avx_detail.h"

#include <immintrin.h>
#include <algorithm>
#include <cassert>

namespace KFP {
namespace SIMD {

// ------------------------------------------------------
// Constructors
// ------------------------------------------------------
template <> inline simd_int::SimdClassBase()
{
    data_.simd_ = _mm256_setzero_si256();
}
// Constructor to broadcast the same value into all elements:
template <> inline simd_int::SimdClassBase(value_type val)
{
    data_.simd_ = Detail::constant<simd_type, value_type>(val);
}
template<>
template<typename T, typename std::enable_if<true, T>::type*>
inline simd_int::SimdClassBase(const simd_type& val_simd)
{
    data_.simd_ = val_simd;
}
template <> inline simd_int::SimdClassBase(const value_type* val)
{
    data_.simd_ = Detail::load<simd_type, value_type>(val);
}
template <> inline simd_int::SimdClassBase(const simd_int& class_simd)
{
    data_.simd_ = class_simd.data_.simd_;
}

// Assignment constructors:
template <> inline simd_int& simd_int::operator=(value_type val)
{
    data_.simd_ = Detail::constant<simd_type, value_type>(val);
    return *this;
}
template<>
template<typename T, typename std::enable_if<true, T>::type*>
inline simd_int& simd_int::operator=(const simd_type& val_simd)
{
    data_.simd_ = val_simd;
    return *this;
}
template <> inline simd_int& simd_int::operator=(const simd_int& class_simd)
{
    if (this != &class_simd) {
        data_.simd_ = class_simd.data_.simd_;
    }
    return *this;
}

// Assignment constructors:
/*template <>
inline simd_int& simd_int::operator=(value_type val)
{
    data_.simd_ = _mm256_set1_epi32(val);
    return *this;
}*/
/*template <> inline simd_int& simd_int::operator=(const simd_int& class_avx)
{
    data_.simd_ = class_avx.data_.simd_;
    return *this;
}*/
// ------------------------------------------------------
// Factory methods
// ------------------------------------------------------
template <> inline simd_int simd_int::iota(value_type start)
{
    simd_int result;
    result.data_.simd_ = _mm256_setr_epi32(start, start + 1, start + 2, start + 3, start + 4, start + 5, start + 6, start + 7);
    return result;
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
// Member function to load from array (unaligned)
template <> inline simd_int& simd_int::load(const value_type* val_ptr)
{
    data_.simd_ = Detail::load<__m256i, value_type>(val_ptr);
    return *this;
}

template <> inline simd_int& simd_int::load_a(const value_type* val_ptr)
{
    data_.simd_ = Detail::load_a<__m256i, value_type>(val_ptr);
    return *this;
}

// Robin: TODO: The SSE equivalent function from Akhil inserts values at the beginning, not at the end.
template <>
inline simd_int& simd_int::load_partial(int n, const value_type* val_ptr)
{
    switch (n) {
    case 0:
        data_.simd_ = _mm256_setzero_si256();
        break;
    case 1:
        data_.simd_ = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, val_ptr[0]);
        break;
    case 2:
        data_.simd_ = _mm256_set_epi32(0, 0, 0, 0, 0, 0, val_ptr[0], val_ptr[1]);
        break;
    case 3:
        data_.simd_ = _mm256_set_epi32(0, 0, 0, 0, 0, val_ptr[0], val_ptr[1], val_ptr[2]);
        break;
    case 4:
        data_.simd_ = _mm256_set_epi32(0, 0, 0, 0, val_ptr[0], val_ptr[1], val_ptr[2], val_ptr[3]);
        break;
    case 5:
        data_.simd_ = _mm256_set_epi32(0, 0, 0, val_ptr[0], val_ptr[1], val_ptr[2], val_ptr[3], val_ptr[4]);
        break;
    case 6:
        data_.simd_ = _mm256_set_epi32(0, 0, val_ptr[0], val_ptr[1], val_ptr[2], val_ptr[3], val_ptr[4], val_ptr[5]);
        break;
    case 7:
        data_.simd_ = _mm256_set_epi32(0, val_ptr[0], val_ptr[1], val_ptr[2], val_ptr[3], val_ptr[4], val_ptr[5], val_ptr[6]);
        break;
    case 8:
    default:
        data_.simd_ = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(val_ptr));
        break;
    }
    return *this;
}
// Member function to store into array (unaligned)
template <> inline void simd_int::store(value_type* val_ptr) const
{
    Detail::store<__m256i, value_type>(data_.simd_, val_ptr);
}

template <> inline void simd_int::store_a(value_type* val_ptr) const
{
    Detail::store_a<__m256i, value_type>(data_.simd_, val_ptr);
}

template <> inline void simd_int::store_stream(value_type* val_ptr) const
{
    _mm256_stream_si256(reinterpret_cast<__m256i*>(val_ptr), data_.simd_);
}

template <>
inline void simd_int::store_partial(int n, value_type* val_ptr) const
{
    if (n < 1)
        return;
    if (n > SimdLen) {
        n = SimdLen;
    }
    value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a(data);
    std::copy_n(data, n, val_ptr);
}
// ------------------------------------------------------
// Gather and Scatter
// ------------------------------------------------------
template <>
inline simd_int& simd_int::gather(const value_type* val_ptr, const simd_int& index)
{
    int __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        indices[__KFP_SIMD__Len_Int]{}; // Helper indices array
    Detail::store_a<__m256i, int>(index.data_.simd_, indices);
    data_.simd_ = _mm256_setr_epi32(
        val_ptr[indices[0]], val_ptr[indices[1]], val_ptr[indices[2]], val_ptr[indices[3]],
        val_ptr[indices[4]], val_ptr[indices[5]], val_ptr[indices[6]], val_ptr[indices[7]]);
    return *this;
}

template <>
inline void simd_int::scatter(value_type* val_ptr, const simd_int& index) const
{
    value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a(data);
    int __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        indices[__KFP_SIMD__Len_Int]{}; // Helper indices array
    Detail::store_a<__m256i, int>(index.data_.simd_, indices);
    val_ptr[indices[0]] = data[0];
    val_ptr[indices[1]] = data[1];
    val_ptr[indices[2]] = data[2];
    val_ptr[indices[3]] = data[3];
    val_ptr[indices[4]] = data[4];
    val_ptr[indices[5]] = data[5];
    val_ptr[indices[6]] = data[6];
    val_ptr[indices[7]] = data[7];
}

// ------------------------------------------------------
// Data member accessors
// ------------------------------------------------------
template <> inline simd_int::value_type simd_int::operator[](int index) const
{
    assert((index >= 0) && "[Error] (operator[]): invalid index. Negative");
    assert((index < SimdLen) && "[Error] (operator[]): invalid index. Exceeds maximum");
    return Detail::extract<value_type, simd_type>(index & 0x07, data_.simd_);
}

// ------------------------------------------------------
// Data elements manipulation
// ------------------------------------------------------
template <> inline simd_int& simd_int::insert(int index, value_type val)
{
    assert((index >= 0) && "[Error] (insert): invalid index. Negative");
    assert((index < SimdLen) && "[Error] (insert): invalid index. Exceeds maximum");
    Detail::insert<simd_type, value_type>(data_.simd_, index & 0x07, val); // Adjusted mask to 0x07 for AVX
    return *this;
}

template <> inline simd_int simd_int::insertCopy(int index, value_type val) const
{
    assert((index >= 0) && "[Error] (insertCopy): invalid index. Negative");
    assert((index < SimdLen) && "[Error] (insertCopy): invalid index. Exceeds maximum");
    simd_int result{*this};
    Detail::insert<simd_type, value_type>(result.data_.simd_, index & 0x07, val); // Adjusted mask to 0x07 for AVX
    return result;
}
template <> inline simd_int& simd_int::cutoff(int n)
{
    value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a(data);
    return load_partial(n, data);
}

template <> inline simd_int simd_int::cutoffCopy(int n) const
{
    value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a(data);
    return simd_int{}.load_partial(n, data);
}

template <> inline simd_int& simd_int::shiftLeft(int n)
{
    data_.simd_ = Detail::shiftLLanes<simd_type>(n, data_.simd_);
    return *this;
}

template <> inline simd_int simd_int::shiftLeftCopy(int n) const
{
    simd_int result;
    result.data_.simd_ = Detail::shiftLLanes<simd_type>(n, data_.simd_);
    return result;
}

template <> inline simd_int& simd_int::shiftRight(int n)
{
    data_.simd_ = Detail::shiftRLanes<simd_type>(n, data_.simd_);
    return *this;
}

template <> inline simd_int simd_int::shiftRightCopy(int n) const
{
    simd_int result;
    result.data_.simd_ = Detail::shiftRLanes<simd_type>(n, data_.simd_);
    return result;
}

template <> inline simd_int& simd_int::rotate(int n)
{
    data_.simd_ = Detail::rotate<simd_type>(n & 0x07, data_.simd_); // Adjusted mask to 0x07 for AVX
    return *this;
}
template <> inline simd_int simd_int::rotateCopy(int n) const
{
    simd_int result;
    result.data_.simd_ = Detail::rotate<simd_type>(n & 0x07, data_.simd_); // Adjusted mask to 0x07 for AVX
    return result;
}

inline simd_int select(const simd_mask& mask, const simd_int& a, const simd_int& b)
{
    return simd_int(
        Detail::select<simd_int::simd_type>(mask.maski(), a.simd(), b.simd()));
}

template <typename F> inline simd_int apply(const simd_int& a, const F& func)
{
    simd_int::value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    a.store_a(data);
    return simd_int{ _mm256_setr_epi32(func(data[0]), func(data[1]), func(data[2]), func(data[3]),
                                       func(data[4]), func(data[5]), func(data[6]), func(data[7])) };
}

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_AVX_IMPL_INT_H
