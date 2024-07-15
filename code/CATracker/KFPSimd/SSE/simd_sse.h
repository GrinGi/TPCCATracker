// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SSE_H
#define SIMD_SSE_H

// Mask
#include "simd_sse_impl_mask.h"
// Int
#include "simd_sse_impl_int.h"
// Float
#include "simd_sse_impl_float.h"

static_assert(
    (KFP::SIMD::simd_float::SimdSize == __KFP_SIMD__SSE_Size_Float),
    "[Error] (simd_sse.h): KFP::SIMD::simd_float given invalid size of simd type.");
static_assert(
    (KFP::SIMD::simd_float::SimdLen == __KFP_SIMD__SSE_Len_Float),
    "[Error] (simd_sse.h): KFP::SIMD::simd_float given invalid length of simd type.");

static_assert(
    (KFP::SIMD::simd_int::SimdSize == __KFP_SIMD__SSE_Size_Int),
    "[Error] (simd_sse.h): KFP::SIMD::simd_int given invalid size of simd type.");
static_assert(
    (KFP::SIMD::simd_int::SimdLen == __KFP_SIMD__SSE_Len_Int),
    "[Error] (simd_sse.h): KFP::SIMD::simd_int given invalid length of simd type.");

#endif // !SIMD_SSE_H
