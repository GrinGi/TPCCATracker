// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_AVX_TYPE_H
#define SIMD_AVX_TYPE_H

#include "../Base/simd_data.h"
#include "../Base/simd_mask.h"
#include "../Base/simd_class.h"

#include <type_traits>

namespace KFP {
namespace SIMD {

using simd_mask = SimdMaskBase<Tag::AVX>;

using simd_float = SimdClassBase<float, Tag::AVX>;
static_assert(std::is_same<simd_float::value_type, float>::value,
              "[Error]: Invalid value type for AVX float SimdClass.");

using simd_int = SimdClassBase<int, Tag::AVX>;
static_assert(std::is_same<simd_int::value_type, int>::value,
              "[Error]: Invalid value type for AVX int SimdClass.");

namespace Detail{

typedef simd_int::simd_type SimdDataI;
typedef simd_float::simd_type SimdDataF;

typedef simd_int::value_type ValueDataI;
typedef simd_float::value_type ValueDataF;

} // namespace Detail

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_AVX_TYPE_H
