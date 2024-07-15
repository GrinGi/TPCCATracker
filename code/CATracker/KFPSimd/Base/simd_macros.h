// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_MACROS_H
#define SIMD_MACROS_H

#include "simd_detect.h"

#define __KFP_SIMD__INLINE inline __attribute__((always_inline))
#define __KFP_SIMD__ATTR_ALIGN(x) __attribute__((aligned(x)))
#define __KFP_SIMD__SPEC_ALIGN(x) alignas(x)

#define __KFP_SIMD__AVX_Size_Int 32
#define __KFP_SIMD__AVX_Len_Int 8
#define __KFP_SIMD__AVX_Size_Float 32
#define __KFP_SIMD__AVX_Len_Float 8

#define __KFP_SIMD__SSE_Size_Int 16
#define __KFP_SIMD__SSE_Len_Int 4
#define __KFP_SIMD__SSE_Size_Float 16
#define __KFP_SIMD__SSE_Len_Float 4

#define __KFP_SIMD__Scalar_Size_Int 4
#define __KFP_SIMD__Scalar_Len_Int 1
#define __KFP_SIMD__Scalar_Size_Float 4
#define __KFP_SIMD__Scalar_Len_Float 1

#if defined(__KFP_SIMD__AVX)
#define __KFP_SIMD__Size_Int __KFP_SIMD__AVX_Size_Int
#define __KFP_SIMD__Len_Int __KFP_SIMD__AVX_Len_Int
#define __KFP_SIMD__Size_Float __KFP_SIMD__AVX_Size_Float
#define __KFP_SIMD__Len_Float __KFP_SIMD__AVX_Len_Float
#elif defined(__KFP_SIMD__SSE)
#define __KFP_SIMD__Size_Int __KFP_SIMD__SSE_Size_Int
#define __KFP_SIMD__Len_Int __KFP_SIMD__SSE_Len_Int
#define __KFP_SIMD__Size_Float __KFP_SIMD__SSE_Size_Float
#define __KFP_SIMD__Len_Float __KFP_SIMD__SSE_Len_Float
#elif defined(__KFP_SIMD__Scalar)
#define __KFP_SIMD__Size_Int __KFP_SIMD__Scalar_Size_Int
#define __KFP_SIMD__Len_Int __KFP_SIMD__Scalar_Len_Int
#define __KFP_SIMD__Size_Float __KFP_SIMD__Scalar_Size_Float
#define __KFP_SIMD__Len_Float __KFP_SIMD__Scalar_Len_Float
#else
#error \
    "[Error] (simd_macros.hpp): Invalid KFParticle SIMD implementation value was selected."
#endif

#endif
