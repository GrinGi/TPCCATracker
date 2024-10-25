// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_AVX_H
#define SIMD_AVX_H

// Mask
#include "simd_avx_impl_mask.h"
// Int
#include "simd_avx_impl_int.h"
// Float
#include "simd_avx_impl_float.h"

static_assert(
    (KFP::SIMD::simd_float::SimdSize == __KFP_SIMD__AVX_Size_Float),
    "[Error] (simd_avx.h): KFP::SIMD::simd_float given invalid size of simd type.");
static_assert(
    (KFP::SIMD::simd_float::SimdLen == __KFP_SIMD__AVX_Len_Float),
    "[Error] (simd_avx.h): KFP::SIMD::simd_float given invalid length of simd type.");

static_assert(
    (KFP::SIMD::simd_int::SimdSize == __KFP_SIMD__AVX_Size_Int),
    "[Error] (simd_avx.h): KFP::SIMD::simd_int given invalid size of simd type.");
static_assert(
    (KFP::SIMD::simd_int::SimdLen == __KFP_SIMD__AVX_Len_Int),
    "[Error] (simd_avx.h): KFP::SIMD::simd_int given invalid length of simd type.");

#endif // !SIMD_AVX_H
