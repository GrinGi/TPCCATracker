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

#include <cassert>

namespace KFP {
namespace SIMD {

template <> __KFP_SIMD__INLINE simd_mask::SimdMaskBase()
{
    mask_ = Detail::constant<simd_typei, value_typei>(0) ;
}
template <> __KFP_SIMD__INLINE simd_mask::SimdMaskBase(bool val)
{
    mask_ = Detail::constant<simd_typei, value_typei>(-int(val)) ;
}
template <> __KFP_SIMD__INLINE simd_mask::SimdMaskBase(const bool* val_ptr)
{

    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_typei
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    data[0] = -int(val_ptr[0]);
    data[1] = -int(val_ptr[1]);
    data[2] = -int(val_ptr[2]);
    data[3] = -int(val_ptr[3]);
    mask_ = Detail::load<simd_typei, value_typei>(data) ;
}
template <> __KFP_SIMD__INLINE simd_mask::SimdMaskBase(const simd_typei& mask)
{
    mask_ = mask;
}
template <> __KFP_SIMD__INLINE simd_mask::SimdMaskBase(const simd_typef& mask)
{
    mask_ = Detail::type_cast<simd_typei, simd_typef>(mask);
}
template <> __KFP_SIMD__INLINE simd_mask::SimdMaskBase(const simd_mask& class_mask)
{
    mask_ = class_mask.mask_ ;
}
// template <> __KFP_SIMD__INLINE simd_mask::SimdMaskBase(const simd_int& mask)
// {
//     mask_ = Detail::NOTBits<simd_typei>( Detail::equal<simd_typei>(mask.simd(), _mm_setzero_si128()) );
// }
// template <> __KFP_SIMD__INLINE simd_mask::SimdMaskBase(const simd_float& mask)
// {
//     mask_ = Detail::NOTBits<simd_typei>( Detail::type_cast<simd_typei, simd_typef>(Detail::equal<simd_typef>(mask.simd(), _mm_setzero_ps())) );
// }

// Assignment operators
template <> __KFP_SIMD__INLINE simd_mask& simd_mask::operator=(bool val)
{
    mask_ = Detail::constant<simd_typei, value_typei>(-int(val)) ;
    return *this ;
}
template <> __KFP_SIMD__INLINE simd_mask& simd_mask::operator=(const simd_typei& mask)
{
    mask_ = mask;
    return *this ;
}
template <> __KFP_SIMD__INLINE simd_mask& simd_mask::operator=(const simd_typef& mask)
{
    mask_ = Detail::type_cast<simd_typei, simd_typef>(mask);
    return *this ;
}
template <> __KFP_SIMD__INLINE simd_mask& simd_mask::operator=(const simd_mask& class_mask)
{
    mask_ = class_mask.mask_ ;
    return *this ;
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE simd_mask& simd_mask::load(const bool* val_ptr)
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_typei
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    data[0] = -int(val_ptr[0]);
    data[1] = -int(val_ptr[1]);
    data[2] = -int(val_ptr[2]);
    data[3] = -int(val_ptr[3]);
    mask_ = Detail::load_a<simd_typei, value_typei>(data) ;
    return *this;
}
template <> __KFP_SIMD__INLINE void simd_mask::store(bool* val_ptr) const
{
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_typei
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
template <> __KFP_SIMD__INLINE int simd_mask::count() const
{
#if 1
    const int tmp{ Detail::sign<value_typei, simd_typei>(mask_) };
    return (tmp & 0x01) + ((tmp & 0x02) >> 1) + ((tmp & 0x04) >> 2) + ((tmp & 0x08) >> 3);
#else
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_typei
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    Detail::store(mask_, data);
    return -(data[0] + data[1] + data[2] + data[3]) ;
#endif
}
template <> __KFP_SIMD__INLINE bool simd_mask::AND() const
{
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
    return _mm_testc_si128(mask_, Detail::getMask<Detail::MASK::TRUE>());
#else
    return Detail::sign<value_typei, simd_typei>(mask_) == 0x0000000F;
#endif
}
template <> __KFP_SIMD__INLINE bool simd_mask::OR() const
{
#if defined(__KFP_SIMD__SSE4_1) // SSE4.1
    return not _mm_testz_si128(mask_, mask_);
#else
    return Detail::sign<value_typei, simd_typei>(mask_) != 0;
#endif
}

// ------------------------------------------------------
// Data member accessors
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE bool simd_mask::operator[](int index) const
{
    return static_cast<bool>(Detail::extract<value_typei, simd_typei>(index, mask_));
}
template <> __KFP_SIMD__INLINE simd_float::simd_type simd_mask::maskf() const
{
    return Detail::type_cast<simd_typef, simd_typei>(mask_);
}

// ------------------------------------------------------
// Data elements manipulation
// ------------------------------------------------------
template <> __KFP_SIMD__INLINE simd_mask& simd_mask::insert(size_t index, bool val)
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
template <> __KFP_SIMD__INLINE simd_mask simd_mask::insertCopy(size_t index, bool val) const
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
template <> __KFP_SIMD__INLINE simd_mask& simd_mask::cutoff(size_t n)
{
    if(n >= SimdLen) return *this;
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_typei
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    Detail::store<simd_typei, value_typei>(mask_, data);
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
    mask_ = Detail::load<simd_typei, value_typei>(mask);
    return *this;
}
template <> __KFP_SIMD__INLINE simd_mask simd_mask::cutoffCopy(size_t n) const
{
    if(n >= SimdLen) return *this;
    __KFP_SIMD__SPEC_ALIGN(__KFP_SIMD__Size_Int) value_typei
    data[__KFP_SIMD__Len_Int]{}; // Helper data array
    Detail::store<simd_typei, value_typei>(mask_, data);
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
    result.mask_ = Detail::load<simd_typei, value_typei>(mask);
    return result;
}

template <>
__KFP_SIMD__INLINE void Detail::print<simd_mask>(std::ostream &stream, const simd_mask &a) {
    bool mask[__KFP_SIMD__Len_Int]{}; // Helper mask array
    a.store(mask);
    stream << "[" << std::boolalpha;
    for(int idx{0} ; idx != (simd_mask::SimdLen-1) ; ++idx){
        stream << mask[idx] << ", ";
    }
    stream << mask[(simd_mask::SimdLen-1)] << std::noboolalpha << "]";
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
