// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SSE_IMPL_FLOAT_H
#define SIMD_SSE_IMPL_FLOAT_H

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
template <> __KFP_SIMD__INLINE simd_float::SimdClassBase()
{
    data_.simd_ = _mm_setzero_ps();
}
// Constructor to broadcast the same value into all elements:
template <> __KFP_SIMD__INLINE simd_float::SimdClassBase(value_type val)
{
    data_.simd_ = Detail::constant<simd_type, value_type>(val);
}
template <>
template <typename T, typename std::enable_if<true, T>::type*>
__KFP_SIMD__INLINE simd_float::SimdClassBase(const simd_type& val_simd)
{
    data_.simd_ = val_simd;
}
template <> __KFP_SIMD__INLINE simd_float::SimdClassBase(const value_type* val)
{
    data_.simd_ = Detail::load<simd_type, value_type>(val);
}
template <> __KFP_SIMD__INLINE simd_float::SimdClassBase(const simd_float& class_simd)
{
    data_.simd_ = class_simd.data_.simd_;
}

// Assignment constructors:
template <>
__KFP_SIMD__INLINE simd_float& simd_float::operator=(value_type val)
{
    data_.simd_ = Detail::constant<simd_type, value_type>(val);
    return *this;
}
template <>
template <typename T, typename std::enable_if<true, T>::type*>
__KFP_SIMD__INLINE simd_float& simd_float::operator=(const simd_type& val_simd)
{
    data_.simd_ = val_simd;
    return *this;
}
template <> __KFP_SIMD__INLINE simd_float& simd_float::operator=(const simd_float& class_simd)
{
    data_.simd_ = class_simd.data_.simd_;
    return *this;
}

// ------------------------------------------------------
// Factory methods
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE simd_float simd_float::iota(value_type start)
{
    simd_float result;
    result.data_.simd_ = _mm_setr_ps(start, start + 1.0f, start + 2.0f, start + 3.0f);
    return result;
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
// Member function to load from array (unaligned)
template <> __KFP_SIMD__INLINE simd_float& simd_float::load(const value_type* val_ptr)
{
    data_.simd_ = Detail::load<simd_type, value_type>(val_ptr);
    return *this;
}
// Member function to load from array (aligned)
template <> __KFP_SIMD__INLINE simd_float& simd_float::load_a(const value_type* val_ptr)
{
    data_.simd_ = Detail::load_a<simd_type, value_type>(val_ptr);
    return *this;
}
// Partial load. Load n elements and set the rest to 0
template <>
__KFP_SIMD__INLINE simd_float& simd_float::load_partial(int index, const value_type* val_ptr)
{
    switch (index) {
    case 0:
        data_.simd_ = _mm_setzero_ps();
        break;
    case 1:
        data_.simd_ = _mm_load_ss(val_ptr);
        break;
    case 2:
        data_.simd_ = _mm_setr_ps(val_ptr[0], val_ptr[1], 0.f, 0.f);
        break;
    case 3:
        data_.simd_ = _mm_setr_ps(val_ptr[0], val_ptr[1], val_ptr[2], 0.f);
        break;
    case 4:
    default:
        load(val_ptr);
        break;
    }
    return *this;
}
// Member function to store into array (unaligned)
template <> __KFP_SIMD__INLINE void simd_float::store(value_type* val_ptr) const
{
    Detail::store<simd_type, value_type>(data_.simd_, val_ptr);
}
// Member function storing into array (aligned)
template <> __KFP_SIMD__INLINE void simd_float::store_a(value_type* val_ptr) const
{
    Detail::store_a<simd_type, value_type>(data_.simd_, val_ptr);
}
// Member function storing to aligned uncached memory (non-temporal store).
template <> __KFP_SIMD__INLINE void simd_float::store_stream(value_type* val_ptr) const
{
    _mm_stream_ps(val_ptr, data_.simd_);
}
// Partial store. Store n elements
template <>
__KFP_SIMD__INLINE void simd_float::store_partial(int n, value_type* val_ptr) const
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Float) value_type
    data[__KFP_SIMD__Len_Float]{}; // Helper data array
    store_a(data);
    std::copy_n(data, n, val_ptr);
}

// ------------------------------------------------------
// Gather and Scatter
// ------------------------------------------------------
template <>
__KFP_SIMD__INLINE simd_float& simd_float::gather(const value_type* val_ptr, const simd_int& index)
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) simd_int::value_type
    indices[__KFP_SIMD__Len_Int]{}; // Helper indices array
    Detail::store_a<simd_int::simd_type, simd_int::value_type>(index.simd(), indices);
    data_.simd_ = _mm_setr_ps(val_ptr[indices[0]], val_ptr[indices[1]],
                              val_ptr[indices[2]], val_ptr[indices[3]]);
    return *this;
}
template <>
__KFP_SIMD__INLINE void simd_float::scatter(value_type* val_ptr, const simd_int& index) const
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Float) value_type
    data[__KFP_SIMD__Len_Float]{}; // Helper data array
    store_a(data);
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) simd_int::value_type
    indices[__KFP_SIMD__Len_Int]{}; // Helper indices array
    Detail::store_a<simd_int::simd_type, simd_int::value_type>(index.simd(), indices);
    val_ptr[indices[0]] = data[0];
    val_ptr[indices[1]] = data[1];
    val_ptr[indices[2]] = data[2];
    val_ptr[indices[3]] = data[3];
}

// ------------------------------------------------------
// Data member accessors
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE simd_float::value_type simd_float::operator[](int index) const
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
template <> __KFP_SIMD__INLINE simd_float& simd_float::insert(int index, value_type val)
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
template <> __KFP_SIMD__INLINE simd_float simd_float::insertCopy(int index, value_type val) const
{
    assert((index > -1) && ("[Error] (insertCopy): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    assert((index < SimdLen) &&
           ("[Error] (insertCopy): invalid index (" + std::to_string(index) +
            ") given. Exceeds maximum")
               .data());
    simd_float result{*this};
    Detail::insert<simd_type, value_type>(result.data_.simd_, index & 0x03, val);
    return result;

}
template <> __KFP_SIMD__INLINE simd_float& simd_float::cutoff(int n)
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Float) value_type
    data[__KFP_SIMD__Len_Float]{}; // Helper data array
    store_a(data);
    return load_partial(n, data);
}
template <> __KFP_SIMD__INLINE simd_float simd_float::cutoffCopy(int n) const
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Float) value_type
    data[__KFP_SIMD__Len_Float]{}; // Helper data array
    store_a(data);
    return simd_float{}.load_partial(n, data);
}
template <> __KFP_SIMD__INLINE simd_float& simd_float::shiftLeft(int n)
{
    data_.simd_ = Detail::shiftLLanes<simd_type>(n, data_.simd_);
    return *this;
}
template <> __KFP_SIMD__INLINE simd_float simd_float::shiftLeftCopy(int n) const
{
    simd_float result;
    result.data_.simd_ = Detail::shiftLLanes<simd_type>(n, data_.simd_);
    return result;
}
template <> __KFP_SIMD__INLINE simd_float& simd_float::shiftRight(int n)
{
    data_.simd_ = Detail::shiftRLanes<simd_type>(n, data_.simd_);
    return *this;
}
template <> __KFP_SIMD__INLINE simd_float simd_float::shiftRightCopy(int n) const
{
    simd_float result;
    result.data_.simd_ = Detail::shiftRLanes<simd_type>(n, data_.simd_);
    return result;
}
template <> __KFP_SIMD__INLINE simd_float& simd_float::rotate(int n)
{
    data_.simd_ = Detail::rotate<simd_type>(n & 0x03, data_.simd_);
    return *this;
}
template <> __KFP_SIMD__INLINE simd_float simd_float::rotateCopy(int n) const
{
    simd_float result;
    result.data_.simd_ = Detail::rotate<simd_type>(n & 0x03, data_.simd_);
    return result;
}

__KFP_SIMD__INLINE simd_float select(const simd_mask& mask, const simd_float& a, const simd_float& b)
{
    return simd_float(
        Detail::select<simd_float::simd_type>(mask.maskf(), a.simd(), b.simd()));
}

template <typename F> __KFP_SIMD__INLINE simd_float apply(const simd_float& a, const F& func)
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Float) simd_float::value_type
    data[__KFP_SIMD__Len_Float]{}; // Helper data array
    a.store_a(data);
    return simd_float{ _mm_setr_ps(func(data[0]), func(data[1]), func(data[2]),
                              func(data[3])) };
}

__KFP_SIMD__INLINE simd_float round(const simd_float& a)
{
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
    return simd_float{ _mm_round_ps(a.simd(), _MM_FROUND_NINT) };
#elif 0
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Float) simd_float::value_type
    data[__KFP_SIMD__Len_Float]{}; // Helper data array
    a.store(data);
    return simd_float{ _mm_setr_ps(std::round(data[0]), std::round(data[1]),
                              std::round(data[2]), std::round(data[3])) };
#else
    simd_int::simd_type tmp = _mm_cvtps_epi32(a.simd()); // convert to integer
    return simd_float{ Detail::value_cast<simd_float::simd_type, simd_int::simd_type>(tmp) }; // convert back to float
#endif
}

__KFP_SIMD__INLINE simd_mask isInf(const simd_float& a)
{
    return simd_mask{ Detail::equal<simd_float::simd_type>(a.simd(), Detail::type_cast<simd_float::simd_type, simd_int::simd_type>(Detail::getMask<Detail::MASK::INF>())) };
}

__KFP_SIMD__INLINE simd_mask isFinite(const simd_float& a)
{
    const simd_float::simd_type mask_inf = Detail::type_cast<simd_float::simd_type, simd_int::simd_type>(Detail::getMask<Detail::MASK::INF>()) ;
    return simd_mask{ Detail::notEqual<simd_float::simd_type>(Detail::ANDBits<simd_float::simd_type>(a.simd(), mask_inf), mask_inf) };
}

__KFP_SIMD__INLINE simd_mask isNaN(const simd_float& a)
{
    return _mm_cmpunord_ps(a.simd(), a.simd());
}

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_SSE_IMPL_FLOAT_H
