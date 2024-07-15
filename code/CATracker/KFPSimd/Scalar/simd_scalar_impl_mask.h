// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SCALAR_IMPL_MASK_H
#define SIMD_SCALAR_IMPL_MASK_H

#include "../Base/simd_mask.h"
#include "simd_scalar_detail_mask.h"
#include "simd_scalar_type.h"

#include <cmath>

namespace KFP {
namespace SIMD {

template <> inline simd_mask::SimdMaskBase()
{
    mask_ = 0 ;
}
template <> inline simd_mask::SimdMaskBase(bool val)
{
    mask_ = -int(val) ;
}
template <> inline simd_mask::SimdMaskBase(const bool* val_ptr)
{
    mask_ = -int(val_ptr[0]) ;
}
template <> inline simd_mask::SimdMaskBase(const simd_typei& mask)
{
    mask_ = -int(static_cast<bool>(mask)) ;
}
template <> inline simd_mask::SimdMaskBase(const simd_typef& mask)
{
    mask_ = -int(static_cast<bool>(mask)) ;
}
template <> inline simd_mask::SimdMaskBase(const simd_mask& class_mask)
{
    mask_ = class_mask.mask_ ;
}

template <> inline simd_mask& simd_mask::operator=(bool val)
{
    mask_ = -int(val) ;
    return *this ;
}
template <> inline simd_mask& simd_mask::operator=(const simd_typei& mask)
{
    mask_ = -int(static_cast<bool>(mask)) ;
    return *this ;
}
template <> inline simd_mask& simd_mask::operator=(const simd_typef& mask)
{
    mask_ = -int(static_cast<bool>(mask)) ;
    return *this ;
}
template <> inline simd_mask& simd_mask::operator=(const simd_mask& class_mask)
{
    mask_ = class_mask.mask_ ;
    return *this ;
}

template <> inline int simd_mask::count() const
{
    return -(mask_) ;
}
template <> inline bool simd_mask::AND() const
{
    return static_cast<bool>(mask_) ;
}
template <> inline bool simd_mask::OR() const
{
    return static_cast<bool>(mask_) ;
}

template <> inline bool simd_mask::operator[](int) const
{
    return mask_;
}
template <> inline simd_float::simd_type simd_mask::maskf() const
{
    return static_cast<simd_float::simd_type>(mask_);
}

} // namespace SIMD
} // namespace KFP

#endif
