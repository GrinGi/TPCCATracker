// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SSE_TYPE_H
#define SIMD_SSE_TYPE_H

#include "../Base/simd_data.h"
#include "../Base/simd_mask.h"
#include "../Base/simd_class.h"

#include <type_traits>

namespace KFP {
namespace SIMD {

using simd_mask = SimdMaskBase<Tag::SSE>;

using simd_float = SimdClassBase<float, Tag::SSE>;
//static_assert(std::is_same<simd_float::simd_type, __m128>::value,	//TODO: switch back asserts
//              "[Error]: Invalid simd type for SSE float SimdClass.");
//static_assert(std::is_same<simd_float::value_type, float>::value,
//              "[Error]: Invalid value type for SSE float SimdClass.");

using simd_int = SimdClassBase<int, Tag::SSE>;
//static_assert(std::is_same<simd_int::simd_type, __m128i>::value,
//              "[Error]: Invalid simd type for SSE int SimdClass.");
//static_assert(std::is_same<simd_int::value_type, int>::value,
//              "[Error]: Invalid value type for SSE int SimdClass.");

namespace Detail{

typedef simd_int::simd_type SimdDataI;
typedef simd_float::simd_type SimdDataF;

typedef simd_int::value_type ValueDataI;
typedef simd_float::value_type ValueDataF;

} // namespace Detail

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_SSE_TYPE_H
