// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_DETAIL_H
#define SIMD_DETAIL_H

#include <iosfwd>

namespace KFP {
namespace SIMD {

namespace Detail {

// ------------------------------------------------------
// General
// ------------------------------------------------------
template <typename T1, typename T2> T1 type_cast(const T2& val_simd);
template <typename T1, typename T2> T1 value_cast(const T2& val_simd);
template <typename T1, typename T2> T1 constant(T2 val);

// ------------------------------------------------------
// Load and Store
// ------------------------------------------------------
template <typename T1, typename T2> T1 load(const T2* val_ptr);
template <typename T1, typename T2> T1 load_a(const T2* val_ptr);
template <typename T1, typename T2> void store(const T1& val_simd, T2* val_ptr);
template <typename T1, typename T2>
void store_a(const T1& val_simd, T2* val_ptr);

// ------------------------------------------------------
// Logical bitwise
// ------------------------------------------------------
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 ANDBits(const T2& val_simd1, const T3& val_simd2);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 ORBits(const T2& val_simd1, const T3& val_simd2);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 XORBits(const T2& val_simd1, const T3& val_simd2);
template <typename T1, typename T2 = T1> T1 NOTBits(const T2& val_simd);

// ------------------------------------------------------
// Comparison
// ------------------------------------------------------
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 equal(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 notEqual(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 lessThan(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 lessThanEqual(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 greaterThan(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 greaterThanEqual(const T2 &a, const T3 &b);

// ------------------------------------------------------
// Manipulate bits
// ------------------------------------------------------
template <typename T1, typename T2 = T1> T1 shiftLBits(const T2 &a, int b);
template <typename T1, typename T2 = T1> T1 shiftRBits(const T2 &a, int b);

// ------------------------------------------------------
// Logical lanewise
// ------------------------------------------------------
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 ANDLanes(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 ORLanes(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 XORLanes(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1> T1 NOTLanes(const T2 &a);

// ------------------------------------------------------
// Manipulate lanes
// ------------------------------------------------------
template <typename T1, typename T2 = T1, typename T3 = T1, typename T4 = T1>
T1 select(const T2 &mask, const T3 &a, const T4 &b);
template <typename T1, typename T2> T1 extract(int index, const T2 &val_simd);
template <typename T1, typename T2>
void insert(T1 &val_simd, int index, T2 val);
template <typename T1, typename T2 = T1>
T1 shiftLLanes(int n, const T2 &val_simd);
template <typename T1, typename T2 = T1>
T1 shiftRLanes(int n, const T2 &val_simd);
template <typename T1, typename T2 = T1> T1 rotate(int n, const T2 &val_simd);

// ------------------------------------------------------
// Basic Arithmetic
// ------------------------------------------------------
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 add(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 substract(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 multiply(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 divide(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1> T1 minus(const T2 &a);

template <typename T1, typename T2 = T1, typename T3 = T1>
T1 min(const T2 &a, const T3 &b);
template <typename T1, typename T2 = T1, typename T3 = T1>
T1 max(const T2 &a, const T3 &b);

template <typename T1, typename T2 = T1> T1 sqrt(const T2 &a);
/* Reciprocal( inverse) Square Root */
template <typename T1, typename T2 = T1> T1 rsqrt(const T2 &a);
template <typename T1, typename T2 = T1> T1 abs(const T2 &a);
template <typename T1, typename T2 = T1> T1 log(const T2 &a);
template <typename T1, typename T2 = T1> T1 pow(const T2 &a, int exp);
template <typename T1, typename T2 = T1> T1 sign(const T2 &a);

template <typename T> void print(std::ostream &stream, const T &val_simd);

///
template <typename T1, typename T2>
T1 min(const T2 &a);

template <typename T1, typename T2>
T1 max(const T2 &a);

} // namespace Detail

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_DETAIL_H
