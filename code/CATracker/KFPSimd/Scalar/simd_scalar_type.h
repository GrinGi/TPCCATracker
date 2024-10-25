// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SCALAR_TYPE_H
#define SIMD_SCALAR_TYPE_H

#include "../Base/simd_data.h"
#include "../Base/simd_mask.h"
#include "../Base/simd_class.h"

#include <type_traits>

namespace KFP {
namespace SIMD {

using simd_mask = SimdMaskBase<Tag::Scalar>;

using simd_float = SimdClassBase<float, Tag::Scalar>;
static_assert(std::is_same<simd_float::simd_type, float>::value,
              "[Error]: Invalid simd type for Scalar float SimdClass.");
static_assert(std::is_same<simd_float::value_type, float>::value,
              "[Error]: Invalid value type for Scalar float SimdClass.");

using simd_int = SimdClassBase<int, Tag::Scalar>;
static_assert(std::is_same<simd_int::simd_type, int>::value,
              "[Error]: Invalid simd type for Scalar int SimdClass.");
static_assert(std::is_same<simd_int::value_type, int>::value,
              "[Error]: Invalid value type for Scalar int SimdClass.");

namespace Detail{

typedef simd_int::simd_type SimdDataI;
typedef simd_float::simd_type SimdDataF;

typedef simd_int::value_type ValueDataI;
typedef simd_float::value_type ValueDataF;

} // namespace Detail

} // namespace SIMD
} // namespace KFP

#endif
