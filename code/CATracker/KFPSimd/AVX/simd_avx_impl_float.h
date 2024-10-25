// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_AVX_IMPL_FLOAT_H
#define SIMD_AVX_IMPL_FLOAT_H

#include "../Base/simd_class.h"
#include "simd_avx_type.h"
#include "simd_avx_detail.h"

#include <x86intrin.h>
#include <algorithm>
#include <cassert>

namespace KFP {
namespace SIMD {

// ------------------------------------------------------
// Constructors
// ------------------------------------------------------
template <> inline simd_float::SimdClassBase()
{
    data_.simd_ = _mm256_setzero_ps();
}
// Constructor to broadcast the same value into all elements:
template <> inline simd_float::SimdClassBase(value_type val)
{
    data_.simd_ = Detail::constant<simd_type, value_type>(val);
}
template <>
template <typename T, typename std::enable_if<true, T>::type*>
inline simd_float::SimdClassBase(const simd_type& val_simd)
{
    data_.simd_ = val_simd;
}
template <> inline simd_float::SimdClassBase(const value_type* val)
{
    data_.simd_ = Detail::load<simd_type, value_type>(val);
}
template <> inline simd_float::SimdClassBase(const simd_float& class_simd)
{
    data_.simd_ = class_simd.data_.simd_;
}

// Assignment constructors:
template <>
inline simd_float& simd_float::operator=(value_type val)
{
    data_.simd_ = Detail::constant<simd_type, value_type>(val);
    return *this;
}
template <>
template <typename T, typename std::enable_if<true, T>::type*>
inline simd_float& simd_float::operator=(const simd_type& val_simd)
{
    data_.simd_ = val_simd;
    return *this;
}
template <> inline simd_float& simd_float::operator=(const simd_float& class_simd)
{
    data_.simd_ = class_simd.data_.simd_;
    return *this;
}

// ------------------------------------------------------
// Factory methods
// ------------------------------------------------------
template <> inline simd_float simd_float::iota(value_type start)
{
    simd_float result;
    result.data_.simd_ = _mm256_setr_ps(start, start + 1.0f, start + 2.0f, start + 3.0f,
                                        start + 4.0f, start + 5.0f, start + 6.0f, start + 7.0f);
    return result;
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
// Member function to load from array (unaligned)
template <> inline simd_float& simd_float::load(const value_type* val_ptr)
{
    data_.simd_ = Detail::load<simd_type, value_type>(val_ptr);
    return *this;
}
// Member function to load from array (aligned)
template <> inline simd_float& simd_float::load_a(const value_type* val_ptr)
{
    data_.simd_ = Detail::load_a<simd_type, value_type>(val_ptr);
    return *this;
}
// Partial load. Load n elements and set the rest to 0
template <> //TODO
inline simd_float& simd_float::load_partial(int index, const value_type* val_ptr)
{
    switch (index) {
    case 0:
        data_.simd_ = _mm256_setzero_ps(); 
        break;
    case 1:
        data_.simd_ = _mm256_setr_ps(val_ptr[0], 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        break;
    case 2:
        data_.simd_ = _mm256_setr_ps(val_ptr[0], val_ptr[1], 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        break;
    case 3:
        data_.simd_ = _mm256_setr_ps(val_ptr[0], val_ptr[1], val_ptr[2], 0.f, 0.f, 0.f, 0.f, 0.f);
        break;
    case 4:
        data_.simd_ = _mm256_setr_ps(val_ptr[0], val_ptr[1], val_ptr[2], val_ptr[3], 0.f, 0.f, 0.f, 0.f);
        break;
    case 5:
        data_.simd_ = _mm256_setr_ps(val_ptr[0], val_ptr[1], val_ptr[2], val_ptr[3], val_ptr[4], 0.f, 0.f, 0.f);
        break;
    case 6:
        data_.simd_ = _mm256_setr_ps(val_ptr[0], val_ptr[1], val_ptr[2], val_ptr[3], val_ptr[4], val_ptr[5], 0.f, 0.f);
        break;
    case 7:
        data_.simd_ = _mm256_setr_ps(val_ptr[0], val_ptr[1], val_ptr[2], val_ptr[3], val_ptr[4], val_ptr[5], val_ptr[6], 0.f);
        break;
    case 8:
    default:
        data_.simd_ = _mm256_loadu_ps(val_ptr);
        break;
    }
    return *this;
}
// Member function to store into array (unaligned)
template <> inline void simd_float::store(value_type* val_ptr) const
{
    _mm256_storeu_ps(val_ptr, data_.simd_);
}

// Member function storing into array (aligned)
template <> inline void simd_float::store_a(value_type* val_ptr) const
{
    _mm256_store_ps(val_ptr, data_.simd_);
}

// Member function storing to aligned uncached memory (non-temporal store).
template <> inline void simd_float::store_stream(value_type* val_ptr) const
{
    _mm256_stream_ps(val_ptr, data_.simd_);
}

// Partial store. Store n elements
template <>
inline void simd_float::store_partial(int n, value_type* val_ptr) const
{
    if (n < 1)
        return;
    if (n > SimdLen) {
        n = SimdLen;
    }
    value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper data array
    store_a(data); // Store to temporary array aligned
    std::copy_n(data, n, val_ptr); // Copy the required number of elements
}

// ------------------------------------------------------
// Gather and Scatter
// ------------------------------------------------------
template <>
inline simd_float& simd_float::gather(const value_type* val_ptr, const simd_int& index)
{
    simd_int::value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        indices[__KFP_SIMD__Len_Int]{}; // Helper indices array
    Detail::store_a<simd_int::simd_type, simd_int::value_type>(index.simd(), indices);
    data_.simd_ = _mm256_setr_ps(val_ptr[indices[0]], val_ptr[indices[1]],
                                 val_ptr[indices[2]], val_ptr[indices[3]],
                                 val_ptr[indices[4]], val_ptr[indices[5]],
                                 val_ptr[indices[6]], val_ptr[indices[7]]);
    return *this;
}
template <>
inline void simd_float::scatter(value_type* val_ptr, const simd_int& index) const
{
    value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper data array
    store_a(data);
    simd_int::value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        indices[__KFP_SIMD__Len_Int]{}; // Helper indices array
    Detail::store_a<simd_int::simd_type, simd_int::value_type>(index.simd(), indices);
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
template <> inline simd_float::value_type simd_float::operator[](int index) const
{
    assert((index > -1) && ("[Error] (operator[]): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    assert((index < SimdLen) &&
           ("[Error] (operator[]): invalid index (" + std::to_string(index) +
            ") given. Exceeds maximum")
               .data());
    return Detail::extract<value_type, simd_type>(index, data_.simd_);
}
// ------------------------------------------------------
// Data elements manipulation
// ------------------------------------------------------
template <> inline simd_float& simd_float::insert(int index, value_type val)
{
    assert((index > -1) && ("[Error] (insert): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    assert((index < SimdLen) &&
           ("[Error] (insert): invalid index (" + std::to_string(index) +
            ") given. Exceeds maximum")
               .data());
    Detail::insert<simd_type, value_type>(data_.simd_, index & 0x07, val); // Corrected bitmask for AVX
    return *this;
}
template <> inline simd_float simd_float::insertCopy(int index, value_type val) const
{
    assert((index > -1) && ("[Error] (insertCopy): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    assert((index < SimdLen) &&
           ("[Error] (insertCopy): invalid index (" + std::to_string(index) +
            ") given. Exceeds maximum")
               .data());
    simd_float result{*this};
    Detail::insert<simd_type, value_type>(result.data_.simd_, index & 0x07, val); // Corrected bitmask for AVX
    return result;
}
template <> inline simd_float& simd_float::cutoff(int n)
{
    value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper data array
    store_a(data);
    return load_partial(n, data);
}
template <> inline simd_float simd_float::cutoffCopy(int n) const
{
    value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper data array
    store_a(data);
    return simd_float{}.load_partial(n, data);
}
template <> inline simd_float& simd_float::shiftLeft(int n)
{
    data_.simd_ = Detail::shiftLLanes<simd_type>(n, data_.simd_);
    return *this;
}
template <> inline simd_float simd_float::shiftLeftCopy(int n) const
{
    simd_float result;
    result.data_.simd_ = Detail::shiftLLanes<simd_type>(n, data_.simd_);
    return result;
}
template <> inline simd_float& simd_float::shiftRight(int n)
{
    data_.simd_ = Detail::shiftRLanes<simd_type>(n, data_.simd_);
    return *this;
}
template <> inline simd_float simd_float::shiftRightCopy(int n) const
{
    simd_float result;
    result.data_.simd_ = Detail::shiftRLanes<simd_type>(n, data_.simd_);
    return result;
}
template <> inline simd_float& simd_float::rotate(int n)
{
    data_.simd_ = Detail::rotate<simd_type>(n & 0x07, data_.simd_); // Adjusted for AVX256
    return *this;
}
template <> inline simd_float simd_float::rotateCopy(int n) const
{
    simd_float result;
    result.data_.simd_ = Detail::rotate<simd_type>(n & 0x07, data_.simd_); // Adjusted for AVX256
    return result;
}

inline simd_float select(const simd_mask& mask, const simd_float& a, const simd_float& b)
{
    return simd_float(
        Detail::select<simd_float::simd_type>(mask.maskf(), a.simd(), b.simd()));
}
template <typename F> inline simd_float apply(const simd_float& a, const F& func)
{
    simd_float::value_type __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Float)
        data[__KFP_SIMD__Len_Float]{}; // Helper data array
    a.store_a(data);
    return simd_float{ _mm256_setr_ps(func(data[0]), func(data[1]), func(data[2]), func(data[3]),
                                      func(data[4]), func(data[5]), func(data[6]), func(data[7])) };
}

inline simd_float round(const simd_float& a)
{
    return simd_float{ _mm256_round_ps(a.simd(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) };
}

inline simd_mask isInf(const simd_float& a)
{
    auto inf_mask = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    return simd_mask{ _mm256_cmp_ps(a.simd(), inf_mask, _CMP_EQ_OQ) };
}

inline simd_mask isFinite(const simd_float& a)
{
    auto inf_mask = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    return simd_mask{ _mm256_cmp_ps(a.simd(), inf_mask, _CMP_NEQ_OQ) };
}

inline simd_mask isNaN(const simd_float& a)
{
    return simd_mask{ _mm256_cmp_ps(a.simd(), a.simd(), _CMP_UNORD_Q) };
}

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_AVX_IMPL_FLOAT_H
