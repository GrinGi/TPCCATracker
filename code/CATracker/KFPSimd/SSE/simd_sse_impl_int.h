// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SSE_IMPL_INT_H
#define SIMD_SSE_IMPL_INT_H

#include "../Base/simd_class.h"
#include "simd_sse_type.h"
#include "simd_sse_detail.h"

#include <algorithm>
#include <cassert>

namespace KFP {
namespace SIMD {

// ------------------------------------------------------
// Constructors
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE simd_int::SimdClassBase()
{
    data_.simd_ = _mm_setzero_si128();
}
// Constructor to broadcast the same value into all elements:
template <> __KFP_SIMD__INLINE simd_int::SimdClassBase(value_type val)
{
    data_.simd_ = Detail::constant<simd_type, value_type>(val);
}
template<>
template<typename T, typename std::enable_if<true, T>::type*>
__KFP_SIMD__INLINE simd_int::SimdClassBase(const simd_type& val_simd)
{
    data_.simd_ = val_simd;
}
template <> __KFP_SIMD__INLINE simd_int::SimdClassBase(const value_type* val)
{
    data_.simd_ = Detail::load<simd_type, value_type>(val);
}
template <> __KFP_SIMD__INLINE simd_int::SimdClassBase(const simd_int& class_simd)
{
    data_.simd_ = class_simd.data_.simd_;
}

// Assignment constructors:
template <>
__KFP_SIMD__INLINE simd_int& simd_int::operator=(value_type val)
{
    data_.simd_ = Detail::constant<simd_type, value_type>(val);
    return *this;
}
template<>
template<typename T, typename std::enable_if<true, T>::type*>
__KFP_SIMD__INLINE simd_int& simd_int::operator=(const simd_type& val_simd)
{
    data_.simd_ = val_simd;
    return *this;
}
template <> __KFP_SIMD__INLINE simd_int& simd_int::operator=(const simd_int& class_simd)
{
    data_.simd_ = class_simd.data_.simd_;
    return *this;
}

// ------------------------------------------------------
// Factory methods
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE simd_int simd_int::iota(value_type start)
{
    simd_int result;
    result.data_.simd_ = _mm_setr_epi32(start, start + 1, start + 2, start + 3);
    return result;
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
// Member function to load from array (unaligned)
template <> __KFP_SIMD__INLINE simd_int& simd_int::load(const value_type* val_ptr)
{
    data_.simd_ = Detail::load<simd_type, value_type>(val_ptr);
    return *this;
}
// Member function to load from array (aligned)
template <> __KFP_SIMD__INLINE simd_int& simd_int::load_a(const value_type* val_ptr)
{
    data_.simd_ = Detail::load_a<simd_type, value_type>(val_ptr);
    return *this;
}
// Partial load. Load n elements and set the rest to 0
template <>
__KFP_SIMD__INLINE simd_int& simd_int::load_partial(int n, const value_type* val_ptr)
{
    switch (n) {
    case 0:
        data_.simd_ = _mm_setzero_si128();
        break;
    case 1:
        // data_.simd_ = _mm_loadu_si32(val_ptr);
        data_.simd_ = _mm_cvtsi32_si128(val_ptr[0]);
        break;
    case 2:
        data_.simd_ = _mm_setr_epi32(val_ptr[0], val_ptr[1], 0, 0);
        break;
    case 3:
        data_.simd_ = _mm_setr_epi32(val_ptr[0], val_ptr[1], val_ptr[2], 0);
        break;
    case 4:
    default:
        load(val_ptr);
        break;
    }
    return *this;
}
// Member function to store into array (unaligned)
template <> __KFP_SIMD__INLINE void simd_int::store(value_type* val_ptr) const
{
    Detail::store<simd_type, value_type>(data_.simd_, val_ptr);
}
// Member function storing into array (aligned)
template <> __KFP_SIMD__INLINE void simd_int::store_a(value_type* val_ptr) const
{
    Detail::store_a<simd_type, value_type>(data_.simd_, val_ptr);
}
// Member function storing to aligned uncached memory (non-temporal store).
template <> __KFP_SIMD__INLINE void simd_int::store_stream(value_type* val_ptr) const
{
    _mm_stream_si128(reinterpret_cast<simd_type*>(val_ptr), data_.simd_);
}
// Partial store. Store n elements
template <>
__KFP_SIMD__INLINE void simd_int::store_partial(int n, value_type* val_ptr) const
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_type
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a(data);
    std::copy_n(data, n, val_ptr);
}

// ------------------------------------------------------
// Gather and Scatter
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE simd_int& simd_int::gather(const value_type* val_ptr, const simd_int& index)
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_type
    indices[__KFP_SIMD__Len_Int]{}; // Helper indices array
    Detail::store_a<simd_type, value_type>(index.data_.simd_, indices);
    data_.simd_ = _mm_setr_epi32( val_ptr[indices[0]], val_ptr[indices[1]], val_ptr[indices[2]], val_ptr[indices[3]]) ;
    return *this;
}
template <>
__KFP_SIMD__INLINE void simd_int::scatter(value_type* val_ptr, const simd_int& index) const
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_type
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a(data);

    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) int
    indices[__KFP_SIMD__Len_Int]{}; // Helper indices array
    Detail::store_a<simd_type, value_type>(index.data_.simd_, indices);

    val_ptr[indices[0]] = data[0];
    val_ptr[indices[1]] = data[1];
    val_ptr[indices[2]] = data[2];
    val_ptr[indices[3]] = data[3];
}

// ------------------------------------------------------
// Data member accessors
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE simd_int::value_type simd_int::operator[](int index) const
{
    assert((index > -1) && ("[Error] (operator[]): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    assert((index < SimdLen) &&
           ("[Error] (operator[]): invalid index (" + std::to_string(index) +
            ") given. Exceeds maximum")
               .data());
    return Detail::extract<value_type, simd_type>(index & 0x03, data_.simd_);
}

// ------------------------------------------------------
// Data elements manipulation
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE simd_int& simd_int::insert(int index, value_type val)
{
    assert((index > -1) && ("[Error] (insert): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    assert((index < SimdLen) &&
           ("[Error] (insert): invalid index (" + std::to_string(index) +
            ") given. Exceeds maximum")
               .data());
    Detail::insert<simd_type, value_type>(data_.simd_, index & 0x03, val);
    return *this;
}
template <> __KFP_SIMD__INLINE simd_int simd_int::insertCopy(int index, value_type val) const
{
    assert((index > -1) && ("[Error] (insertCopy): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    assert((index < SimdLen) &&
           ("[Error] (insertCopy): invalid index (" + std::to_string(index) +
            ") given. Exceeds maximum")
               .data());
    simd_int result{*this};
    Detail::insert<simd_type, value_type>(result.data_.simd_, index & 0x03, val);
    return result;
}
template <> __KFP_SIMD__INLINE simd_int& simd_int::cutoff(int n)
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_type
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a(data);
    return load_partial(n, data);
}
template <> __KFP_SIMD__INLINE simd_int simd_int::cutoffCopy(int n) const
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_type
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    store_a(data);
    return simd_int{}.load_partial(n, data);
}
template <> __KFP_SIMD__INLINE simd_int& simd_int::shiftLeft(int n)
{
    data_.simd_ = Detail::shiftLLanes<simd_type>(n, data_.simd_);
    return *this;
}
template <> __KFP_SIMD__INLINE simd_int simd_int::shiftLeftCopy(int n) const
{
    simd_int result;
    result.data_.simd_ = Detail::shiftLLanes<simd_type>(n, data_.simd_);
    return result;
}
template <> __KFP_SIMD__INLINE simd_int& simd_int::shiftRight(int n)
{
    data_.simd_ = Detail::shiftRLanes<simd_type>(n, data_.simd_);
    return *this;
}
template <> __KFP_SIMD__INLINE simd_int simd_int::shiftRightCopy(int n) const
{
    simd_int result;
    result.data_.simd_ = Detail::shiftRLanes<simd_type>(n, data_.simd_);
    return result;
}
template <> __KFP_SIMD__INLINE simd_int& simd_int::rotate(int n)
{
    data_.simd_ = Detail::rotate<simd_type>(n & 0x03, data_.simd_);
    return *this;
}
template <> __KFP_SIMD__INLINE simd_int simd_int::rotateCopy(int n) const
{
    simd_int result;
    result.data_.simd_ = Detail::rotate<simd_type>(n & 0x03, data_.simd_);
    return result;
}

__KFP_SIMD__INLINE simd_int select(const simd_mask& mask, const simd_int& a, const simd_int& b)
{
    return simd_int(
        Detail::select<simd_int::simd_type>(mask.maski(), a.simd(), b.simd()));
}
template <typename F> __KFP_SIMD__INLINE simd_int apply(const simd_int& a, const F& func)
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) simd_int::value_type
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    a.store(data);
    return simd_int{ _mm_setr_epi32(func(data[0]), func(data[1]), func(data[2]),
                              func(data[3])) };
}

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_SSE_IMPL_INT_H
