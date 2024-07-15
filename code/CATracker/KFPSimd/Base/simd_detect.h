// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_DETECT_H
#define SIMD_DETECT_H

#include <x86intrin.h>

#ifndef __KFP_SIMD__

#if defined(__AVX2__)
#define __KFP_SIMD__AVX2 1
#elif defined(__AVX__)
#define __KFP_SIMD__AVX 1
#else
    #if defined(__SSE4_2__)
    #define __KFP_SIMD__SSE4_2 1
    #define __KFP_SIMD__SSE 1
    #elif defined(__SSE4_1__)
    #define __KFP_SIMD__SSE4_1 1
    #define __KFP_SIMD__SSE 1
    #elif defined(__SSSE3__)
    #define __KFP_SIMD__SSSE3 1
    #define __KFP_SIMD__SSE 1
    #elif defined(__SSE3__)
    #define __KFP_SIMD__SSE3 1
    #define __KFP_SIMD__SSE 1
    #elif defined(__SSE2__)
    #define __KFP_SIMD__SSE2 1
    #define __KFP_SIMD__SSE 1
    #else
    #define __KFP_SIMD__Scalar 1
    #endif
#endif

#else // __KFP_SIMD__

#if __KFP_SIMD__ > 6 // AVX2 supersedes SSE
#define __KFP_SIMD__AVX2 1
#define __KFP_SIMD__AVX 1
#elif __KFP_SIMD__ > 5 // AVX supersedes SSE
#define __KFP_SIMD__AVX 1
#elif __KFP_SIMD__ > 4
#define __KFP_SIMD__SSE4_2 1
#define __KFP_SIMD__SSE 1
#elif __KFP_SIMD__ > 3
#define __KFP_SIMD__SSE4_1 1
#define __KFP_SIMD__SSE 1
#elif __KFP_SIMD__ > 2
#define __KFP_SIMD__SSSE3 1
#define __KFP_SIMD__SSE 1
#elif __KFP_SIMD__ > 1
#define __KFP_SIMD__SSE3 1
#define __KFP_SIMD__SSE 1
#elif __KFP_SIMD__ > 0
#define __KFP_SIMD__SSE2 1
#define __KFP_SIMD__SSE 1
#else
#define __KFP_SIMD__Scalar 1
#endif
#undef __KFP_SIMD__
#endif // __KFP_SIMD__

#if defined(__KFP_SIMD__AVX2)
#define __KFP_SIMD__AVX 1
#endif

#if defined(__KFP_SIMD__AVX)
#define __KFP_SIMD__SSE4_2 1
#endif

#if defined(__KFP_SIMD__SSE4_2)
#define __KFP_SIMD__SSE4_1 1
#endif

#if defined(__KFP_SIMD__SSE4_1)
#define __KFP_SIMD__SSSE3 1
#endif

#if defined(__KFP_SIMD__SSSE3)
#define __KFP_SIMD__SSE3 1
#endif

#if defined(__KFP_SIMD__SSE3)
#define __KFP_SIMD__SSE2 1
#endif

#if defined(__KFP_SIMD__SSE2)
#define __KFP_SIMD__SSE 1
#endif

#if !defined(__KFP_SIMD__Scalar) && !defined(__KFP_SIMD__SSE) && !defined(__KFP_SIMD__AVX)
#error \
    "[Error] (simd_detect.hpp): Invalid KFParticle SIMD implementation value was selected."
#endif
#if defined(__KFP_SIMD__SSE) && !defined(__KFP_SIMD__SSE2)
#error "[Error] (simd_detect.hpp): KFParticle SIMD implementation value SSE selected without SSE2."
#endif

#endif
// vim: foldmethod=marker
