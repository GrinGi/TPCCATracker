// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SSE_IMPL_MASK_H
#define SIMD_SSE_IMPL_MASK_H

#include "../Base/simd_mask.h"
#include "simd_sse_detail.h"
#include "simd_sse_type.h"

#include <x86intrin.h>
#include <cassert>

namespace KFP {
namespace SIMD {

template <> inline simd_mask::SimdMaskBase(bool val)
{
    mask_ = Detail::constant<simd_typei, value_typei>(-int(val)) ;
}
template <> inline simd_mask::SimdMaskBase(const bool* val_ptr)
{
    value_typei __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    data[0] = -int(val_ptr[0]);
    data[1] = -int(val_ptr[1]);
    data[2] = -int(val_ptr[2]);
    data[3] = -int(val_ptr[3]);
    mask_ = Detail::load_a<simd_typei, value_typei>(data) ;
}
template <> inline simd_mask::SimdMaskBase(const simd_typei& mask)
{
    mask_ = mask;
}
template <> inline simd_mask::SimdMaskBase(const simd_typef& mask)
{
    mask_ = Detail::type_cast<simd_typei, simd_typef>(mask);
}
template <> inline simd_mask::SimdMaskBase(const simd_mask& class_mask)
{
    mask_ = class_mask.mask_ ;
}
// template <> inline simd_mask::SimdMaskBase(const simd_int& mask)
// {
//     mask_ = Detail::NOTBits<simd_typei>( Detail::equal<simd_typei>(mask.simd(), _mm_setzero_si128()) );
// }
// template <> inline simd_mask::SimdMaskBase(const simd_float& mask)
// {
//     mask_ = Detail::NOTBits<simd_typei>( Detail::type_cast<simd_typei, simd_typef>(Detail::equal<simd_typef>(mask.simd(), _mm_setzero_ps())) );
// }

// Assignment operators
template <> inline simd_mask& simd_mask::operator=(bool val)
{
    mask_ = Detail::constant<simd_typei, value_typei>(-int(val)) ;
    return *this ;
}
template <> inline simd_mask& simd_mask::operator=(const simd_typei& mask)
{
    mask_ = mask;
    return *this ;
}
template <> inline simd_mask& simd_mask::operator=(const simd_typef& mask)
{
    mask_ = Detail::type_cast<simd_typei, simd_typef>(mask);
    return *this ;
}
template <> inline simd_mask& simd_mask::operator=(const simd_mask& class_mask)
{
    mask_ = class_mask.mask_ ;
    return *this ;
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
template <> inline simd_mask& simd_mask::load(const bool* val_ptr)
{
    value_typei __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    data[0] = -int(val_ptr[0]);
    data[1] = -int(val_ptr[1]);
    data[2] = -int(val_ptr[2]);
    data[3] = -int(val_ptr[3]);
    mask_ = Detail::load_a<simd_typei, value_typei>(data) ;
    return *this;
}
template <> inline void simd_mask::store(bool* val_ptr) const
{
    value_typei __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    Detail::store_a<simd_typei, value_typei>(mask_, data);
    val_ptr[0] = data[0];
    val_ptr[1] = data[1];
    val_ptr[2] = data[2];
    val_ptr[3] = data[3];
}

// ------------------------------------------------------
// Status accessors
// ------------------------------------------------------
template <> inline int simd_mask::count() const
{
#if 1
    const int tmp{ Detail::sign<value_typei, simd_typei>(mask_) };
    return (tmp & 0x01) + ((tmp & 0x02) >> 1) + ((tmp & 0x04) >> 2) + ((tmp & 0x08) >> 3);
#else
    value_typei __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    Detail::store_a(mask_, data);
    return -(data[0] + data[1] + data[2] + data[3]) ;
#endif
}
template <> inline bool simd_mask::AND() const
{
    return static_cast<bool>(count() == 4) ;
}
template <> inline bool simd_mask::OR() const
{
    return static_cast<bool>(count() != 0) ;
}

// ------------------------------------------------------
// Data member accessors
// ------------------------------------------------------
template <> inline bool simd_mask::operator[](int index) const
{
    return static_cast<bool>(Detail::extract<value_typei, simd_typei>(index, mask_));
}
template <> inline simd_float::simd_type simd_mask::maskf() const
{
    return Detail::type_cast<simd_typef, simd_typei>(mask_);
}

// ------------------------------------------------------
// Data elements manipulation
// ------------------------------------------------------
template <> inline simd_mask& simd_mask::insert(size_t index, bool val)
{
    assert((index > -1) && ("[Error] (insert): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    assert((index < SimdLen) &&
           ("[Error] (insert): invalid index (" + std::to_string(index) +
            ") given. Exceeds maximum")
               .data());
    Detail::insert<simd_typei, value_typei>(mask_, index & 0x03, -int(val));
    return *this;
}
template <> inline simd_mask simd_mask::insertCopy(size_t index, bool val) const
{
    assert((index > -1) && ("[Error] (insertCopy): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    assert((index < SimdLen) &&
           ("[Error] (insertCopy): invalid index (" + std::to_string(index) +
            ") given. Exceeds maximum")
               .data());
    simd_mask result{*this};
    Detail::insert<simd_typei, value_typei>(result.mask_, index & 0x03, -int(val));
    return result;
}
template <> inline simd_mask& simd_mask::cutoff(size_t n)
{
    if(n >= SimdLen) return *this;
    value_typei __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    Detail::store_a<simd_typei, value_typei>(mask_, data);
    value_typei __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        mask[__KFP_SIMD__Len_Int]{0, 0, 0, 0}; // Helper mask array
    switch(n){
        case 1:
            mask[0] = data[0];
        break;
        case 2:
            mask[0] = data[0];
            mask[1] = data[1];
        break;
        case 3:
            mask[0] = data[0];
            mask[1] = data[1];
            mask[2] = data[2];
        break;
        case 4:
            mask[0] = data[0];
            mask[1] = data[1];
            mask[2] = data[2];
            mask[3] = data[3];
        break;
        case 0:
        default:
            break;
    }
    mask_ = Detail::load_a<simd_typei, value_typei>(mask);
    return *this;
}
template <> inline simd_mask simd_mask::cutoffCopy(size_t n) const
{
    if(n >= SimdLen) return *this;
    value_typei __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        data[__KFP_SIMD__Len_Int]{}; // Helper data array
    Detail::store_a<simd_typei, value_typei>(mask_, data);
    value_typei __KFP_SIMD__ATTR_ALIGN(__KFP_SIMD__Size_Int)
        mask[__KFP_SIMD__Len_Int]{0, 0, 0, 0}; // Helper mask array
    switch(n){
        case 1:
            mask[0] = data[0];
        break;
        case 2:
            mask[0] = data[0];
            mask[1] = data[1];
        break;
        case 3:
            mask[0] = data[0];
            mask[1] = data[1];
            mask[2] = data[2];
        break;
        case 4:
            mask[0] = data[0];
            mask[1] = data[1];
            mask[2] = data[2];
            mask[3] = data[3];
        break;
        case 0:
        default:
            break;
    }
    simd_mask result;
    result.mask_ = Detail::load_a<simd_typei, value_typei>(mask);
    return result;
}

///
template <> inline bool simd_mask::isEmpty() const	//TODO: test it!
{
#if 0
    __m128i mask = _mm_cmpeq_epi32(mask_, _mm_setzero_si128());
    int maskValue = _mm_movemask_ps(_mm_castsi128_ps(mask));
    return maskValue == 0xF;
#else
    return static_cast<bool>(count() == 0);
#endif
}

template <> inline bool simd_mask::isFull() const
{
  return static_cast<bool>(count() == 4);
}

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_SSE_IMPL_MASK_H
