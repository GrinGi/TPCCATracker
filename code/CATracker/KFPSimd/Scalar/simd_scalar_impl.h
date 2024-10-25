// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SCALAR_IMPL_H
#define SIMD_SCALAR_IMPL_H

#include "../Base/simd_class.h"
#include "simd_scalar_type.h"
#include "simd_scalar_detail.h"

#include <cassert>

namespace KFP {
namespace SIMD {

template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>::SimdClassBase()
{
    data_.simd_ = ValueType{ 0 };
}
template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>::SimdClassBase(ValueType val)
{
    data_.simd_ = val;
}
template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>::SimdClassBase(const SimdClassBase& class_simd)
{
    data_.simd_ = class_simd.data_.simd_;
}

template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>& SimdClassBase<ValueType, tag>::operator=(ValueType val)
{
    data_.simd_ = val;
    return *this;
}
template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>& SimdClassBase<ValueType, tag>::operator=(const SimdClassBase& class_simd)
{
    data_.simd_ = class_simd.data_.simd_;
    return *this;
}

template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag> SimdClassBase<ValueType, tag>::iota(ValueType start)
{
    return start;
}

template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>& SimdClassBase<ValueType, tag>::load(const ValueType* p)
{
    data_.simd_ = *p;
    return *this;
}
template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>& SimdClassBase<ValueType, tag>::load_a(const ValueType* p)
{
    data_.simd_ = *p;
    return *this;
}
template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>& SimdClassBase<ValueType, tag>::load_partial(int index, const ValueType* p)
{
    assert((index > 0) && ("[Error] (load_partial): invalid index (" +
                           std::to_string(index) + ") given. Not postive.")
                              .data());
    if (index < 1)
        return *this;
    data_.simd_ = *p;
    return *this;
}

template<typename ValueType, Tag tag>
void SimdClassBase<ValueType, tag>::store(ValueType* p) const
{
    *p = data_.simd_;
}
template<typename ValueType, Tag tag>
void SimdClassBase<ValueType, tag>::store_a(ValueType* p) const
{
    *p = data_.simd_;
}
template<typename ValueType, Tag tag>
void SimdClassBase<ValueType, tag>::store_stream(ValueType* p) const
{
    *p = data_.simd_;
}
template<typename ValueType, Tag tag>
void SimdClassBase<ValueType, tag>::store_partial(int index, ValueType* p) const
{
    assert((index > 0) && ("[Error] (store_partial): invalid index (" +
                           std::to_string(index) + ") given. Not positive.")
                              .data());
    if (index < 1)
        return;
    *p = data_.simd_;
}

template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>& SimdClassBase<ValueType, tag>::gather(const ValueType* p, const SimdClassBase<int, tag>& indices)
{
    data_.simd_ = p[indices[0]];
    return *this;
}
template<typename ValueType, Tag tag>
void SimdClassBase<ValueType, tag>::scatter(ValueType* p, const SimdClassBase<int, tag>& indices) const
{
    p[indices[0]] = data_.simd_;
}

template<typename ValueType, Tag tag>
ValueType SimdClassBase<ValueType, tag>::operator[](int index) const
{
    assert((index == 0) &&
           ("[Error] (operator[]): invalid index (" +
            std::to_string(index) + ") given. Non-zero index not allowed.")
               .data());
    if (index != 0) {
        std::cerr
            << "[Error] (operator[]): invalid element access. Non zero index for Scalar simd type.\n";
        exit(1);
    }
    return data_.simd_;
}

template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>& SimdClassBase<ValueType, tag>::insert(int index, ValueType val)
{
    assert((index == 0) &&
           ("[Error] (insert): invalid index (" + std::to_string(index) +
            ") given. Non-zero index not allowed.")
               .data());
    if (index != 0)
        return *this;
    data_.simd_ = val;
    return *this;
}
template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag> SimdClassBase<ValueType, tag>::insertCopy(int index, ValueType val) const
{
    assert((index == 0) &&
           ("[Error] (insertCopy): invalid index (" + std::to_string(index) +
            ") given. Non-zero index not allowed.")
               .data());
    if (index != 0)
        return *this;
    SimdClassBase<ValueType, tag> result{*this};
    result.data_.simd_ = val;
    return result;
}
template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag>& SimdClassBase<ValueType, tag>::cutoff(int index)
{
    assert((index > -1) && ("[Error] (cutoff): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    if (index == 0){
        data_.simd_ = ValueType{ 0 };
    }
    return *this;
}
template<typename ValueType, Tag tag>
SimdClassBase<ValueType, tag> SimdClassBase<ValueType, tag>::cutoffCopy(int index) const
{
    assert((index > -1) && ("[Error] (cutoffCopy): invalid index (" +
                            std::to_string(index) + ") given. Negative")
                               .data());
    if (index == 0){
        return SimdClassBase<ValueType, tag>{ 0 };
    }
    return *this;
}
// TODO: add shifts and rotates

template <typename T>
inline T select(const simd_mask& mask, const T& a, const T& b)
{
    return T{(mask.AND() ? a : b)};
}

template <typename T, typename F> inline T apply(const T& a, const F& func)
{
    return T{func(a.simd())};
}

inline simd_float round(const simd_float& a)
{
    return simd_float{std::round(a.simd())};
}

inline simd_mask isInf(const simd_float& a)
{
    return simd_mask{std::isinf(a.simd())};
}

inline simd_mask isFinite(const simd_float& a)
{
    return simd_mask{std::isfinite(a.simd())};
}

inline simd_mask isNaN(const simd_float& a)
{
    return simd_mask{std::isnan(a.simd())};
}

} // namespace SIMD
} // namespace KFP

#endif
