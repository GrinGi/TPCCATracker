// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SCALAR_H
#define SIMD_SCALAR_H

#include "simd_scalar_impl_mask.h"

#include "simd_scalar_impl_index.h"

#include "simd_scalar_impl.h"

static_assert(
    (KFP::SIMD::simd_float::SimdSize == __KFP_SIMD__Scalar_Size_Float),
    "[Error] (simd_scalar.h): KFP::SIMD::simd_float given invalid size of simd type.");
static_assert(
    (KFP::SIMD::simd_float::SimdLen == __KFP_SIMD__Scalar_Len_Float),
    "[Error] (simd_scalar.h): KFP::SIMD::simd_float given invalid length of simd type.");

static_assert(
    (KFP::SIMD::simd_int::SimdSize == __KFP_SIMD__Scalar_Size_Int),
    "[Error] (simd_scalar.h): KFP::SIMD::simd_int given invalid size of simd type.");
static_assert(
    (KFP::SIMD::simd_int::SimdLen == __KFP_SIMD__Scalar_Len_Int),
    "[Error] (simd_scalar.h): KFP::SIMD::simd_int given invalid length of simd type.");

#endif
