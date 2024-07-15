// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_SCALAR_DETAIL_INT_H
#define SIMD_SCALAR_DETAIL_INT_H

#include "../Base/simd_detail.h"
#include "simd_scalar_type.h"

#include <iostream>
#include <cmath>

namespace KFP {
namespace SIMD {

namespace Detail {

template <typename T1, typename T2> T1 cast(const T2& val_simd)
{
    return static_cast<T1>(val_simd);
}

template <typename T1, typename T2> inline T1 rotated(int amount, const T2& val)
{
    return val;
}

template <typename T> void print(std::ostream& stream, const T& val_simd)
{
    stream << "[" << val_simd << "]";
}

template <typename T1, typename T2> T1 minus(const T2& a)
{
    return -a;
}

template <typename T1, typename T2, typename T3>
T1 add(const T2& a, const T3& b)
{
    return (a + b);
}

template <typename T1, typename T2, typename T3>
T1 substract(const T2& a, const T3& b)
{
    return (a - b);
}

template <typename T1, typename T2, typename T3>
T1 multiply(const T2& a, const T3& b)
{
    return (a * b);
}

template <typename T1, typename T2, typename T3>
T1 divide(const T2& a, const T3& b)
{
    return (a / b);
}

template <typename T1, typename T2, typename T3>
T1 min(const T2& a, const T3& b)
{
    return std::min(a, b);
}

template <typename T1, typename T2, typename T3>
T1 max(const T2& a, const T3& b)
{
    return std::max(a, b);
}

template <typename T1, typename T2> T1 sqrt(const T2& a)
{
    return std::sqrt(a);
}

/* Reciprocal( inverse) Square Root */
template <typename T1, typename T2> T1 rsqrt(const T2& a)
{
    return T1{1}/std::sqrt(a) ;
}

template <typename T1, typename T2> T1 abs(const T2& a)
{
    return std::abs(a) ;
}

template <typename T1, typename T2> T1 log(const T2& a)
{
    return std::log(a) ;
}

template <typename T1, typename T2> T1 pow(const T2& a, int exp)
{
    return std::pow(a, exp) ;
}

/* Logical */
template <typename T1, typename T2>
T1 opShiftLeft(const T2& a, int b)
{
    return (a << b) ;
}
template <typename T1, typename T2>
T1 opShiftRight(const T2& a, int b)
{
    return (a >> b) ;
}
template <typename T1, typename T2, typename T3>
T1 opANDbitwise(const T2& a, const T3& b)
{
    return (a&b) ;
}
template <typename T1, typename T2, typename T3>
T1 opORbitwise(const T2& a, const T3& b)
{
    return (a|b) ;
}
template <typename T1, typename T2, typename T3>
T1 opXORbitwise(const T2& a, const T3& b)
{
    return (a^b) ;
}
template <typename T1, typename T2, typename T3>
T1 opAND(const T2& a, const T3& b)
{
    return (a&&b) ;
}
template <typename T1, typename T2, typename T3>
T1 opOR(const T2& a, const T3& b)
{
    return (a||b) ;
}

/* Comparison */
template <typename T1, typename T2, typename T3>
T1 opLessThan(const T2& a, const T3& b)
{
    return static_cast<T1>(a < b) ;
}

template <typename T1, typename T2, typename T3>
T1 opLessThanEqual(const T2& a, const T3& b)
{
    return static_cast<T1>(a <= b) ;
}

template <typename T1, typename T2, typename T3>
T1 opGreaterThan(const T2& a, const T3& b)
{
    return static_cast<T1>(a > b) ;
}

template <typename T1, typename T2, typename T3>
T1 opGreaterThanEqual(const T2& a, const T3& b)
{
    return static_cast<T1>(a >= b) ;
}

template <typename T1, typename T2, typename T3>
T1 opEqual(const T2& a, const T3& b)
{
    return static_cast<T1>(a == b) ;
}

template <typename T1, typename T2, typename T3>
T1 opNotEqual(const T2& a, const T3& b)
{
    return static_cast<T1>(a != b) ;
}

} // namespace Detail

} // namespace SIMD
} // namespace KFP

#endif
