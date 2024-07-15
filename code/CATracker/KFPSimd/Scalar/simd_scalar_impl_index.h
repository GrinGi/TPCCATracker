// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SCALAR_IMPL_INDEX_H
#define SIMD_SCALAR_IMPL_INDEX_H

#include "../Base/simd_index.h"
#include "simd_scalar_type.h"
#include "simd_scalar_detail_index.h"

#include <iostream>
#include <cassert>

namespace KFP {
namespace SIMD {

template<> inline simd_index::SimdIndexBase()
{
    index_ = 0;
}
template<> inline simd_index::SimdIndexBase(int val)
{
    index_ = val;
}
template<> inline simd_index::SimdIndexBase(int* val_ptr)
{
    index_ = val_ptr[0];
}
template<> inline simd_index::SimdIndexBase(const simd_index& class_indices)
{
    index_ = class_indices.index_;
}
template<> inline simd_index::SimdIndexBase(const simd_float::simd_type& val_simd)
{
    index_ = int(val_simd);
}
template<>
inline simd_index::SimdIndexBase(const simd_int& class_simd)
{
    index_ = class_simd.simd();
}
template<> inline simd_index::SimdIndexBase(const simd_float& class_simd)
{
    index_ = static_cast<int>(class_simd.simd());
}

template<> inline simd_index& simd_index::operator=(int val)
{
    index_ = val;
    return *this;
}
template<> inline simd_index& simd_index::operator=(const simd_index& class_indices)
{
    index_ = class_indices.index_;
    return *this;
}
template<> inline simd_index& simd_index::operator=(const simd_float::simd_type& val_simd)
{
    index_ = int(val_simd);
    return *this;
}

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
// Load
template <> inline simd_index& simd_index::load(const int* val_ptr)
{
    index_ = val_ptr[0];
    return *this;
}
template <> inline simd_index& simd_index::load_a(const int* val_ptr)
{
    index_ = val_ptr[0];
    return *this;
}
// Store
template <> inline void simd_index::store(int* val_ptr) const
{
    val_ptr[0] = index_;
}
template <> inline void simd_index::store_a(int* val_ptr) const
{
    val_ptr[0] = index_;
}
template <> inline void simd_index::store_stream(int* val_ptr) const
{
    val_ptr[0] = index_;
}
template<> inline simd_int::value_type simd_index::operator[](int index) const
{
    assert((index == 0) && ("[Error] (operator[]): invalid index (" +
                            std::to_string(index) + ") given. Only zero is allowed.")
                               .data());
    return index_;
}

} // namespace SIMD
} // namespace KFP

#endif
