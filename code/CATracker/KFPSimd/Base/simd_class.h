// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_CLASS_H
#define SIMD_CLASS_H

#include "simd_tag.h"
#include "simd_data.h"
#include "simd_detail.h"

#include <iosfwd>
#include <type_traits>

namespace KFP {
namespace SIMD {

template <Tag tag> class SimdMaskBase;

template <typename ValueType, Tag tag> class SimdClassBase
{
public:
    // static_assert(validateTag(tag), "[Error]: KFP::SIMD::SimdClassBase given invalid tag.") ;
    static_assert(
        std::is_arithmetic<ValueType>::value,
        "[Error] (KFP::SIMD::SimdClassBase): Invalid ValueType provided. Only works for arithmetic ValueTypes.");
    static_assert(
        std::is_same<ValueType, float>::value ||
            std::is_same<ValueType, int>::value,
        "[Error] (KFP::SIMD::SimdClassBase): Invalid ValueType provided. Only works for int and float types.");

    typedef ValueType value_type;
    typedef typename SimdData<ValueType, tag>::simd_type simd_type;
    static constexpr Tag tag_val{ tag };
    static constexpr size_t SimdSize{ sizeof(simd_type) };
    static constexpr size_t SimdLen{ SimdSize / sizeof(ValueType) };

    // ------------------------------------------------------
    // Constructors
    // ------------------------------------------------------
    // Default constructor:
    SimdClassBase();
    // Constructor to broadcast the same value into all elements:
    SimdClassBase(ValueType val);
    template <typename T = void,
              typename std::enable_if<(tag != Tag::Scalar), T>::type* = nullptr>
    SimdClassBase(const simd_type& val_simd);
    SimdClassBase(const ValueType* val_ptr);
    SimdClassBase(const SimdClassBase& class_simd);
    template <typename T,
              typename std::enable_if<!std::is_same<T, ValueType>::value>::type* = nullptr>
    SimdClassBase(const SimdClassBase<T, tag>& class_simd)
    {
        data_.simd_ = Detail::value_cast<simd_type, typename SimdClassBase<T, tag>::simd_type>(class_simd.simd());
    }

    // Assignment constructors:
    SimdClassBase& operator=(ValueType val);
    template <typename T = void,
              typename std::enable_if<(tag != Tag::Scalar), T>::type* = nullptr>
    SimdClassBase& operator=(const simd_type& val_simd);
    SimdClassBase& operator=(const SimdClassBase& class_simd);

    // ------------------------------------------------------
    // Factory methods
    // ------------------------------------------------------
    static SimdClassBase iota(ValueType start);
    static SimdClassBase seq(ValueType start)
    {
        return iota(start);
    }
    template<typename T>
    static SimdClassBase type_cast(const SimdClassBase<T, tag>& a)
    {
        using from_simd_type = typename SimdClassBase<T, tag>::simd_type;
        return SimdClassBase{ Detail::type_cast<simd_type, from_simd_type>(a.simd()) };
    }

    // ------------------------------------------------------
    // Load and Store
    // ------------------------------------------------------
    // Member function to load from array (unaligned)
    SimdClassBase& load(const ValueType* val_ptr);
    // Member function to load from array (aligned)
    SimdClassBase& load_a(const ValueType* val_ptr);
    // Partial load. Load n elements and set the rest to 0
    SimdClassBase& load_partial(int n, const ValueType* val_ptr);

    // Member function to store into array (unaligned)
    void store(ValueType* val_ptr) const;
    // Member function storing into array (aligned)
    // "store_a" is faster than "store" on older Intel processors (Pentium 4, Pentium M, Core 1,
    // Merom, Wolfdale) and Atom, but not on other processors from Intel, AMD or VIA.
    // You may use store_a instead of store if you are certain that val_ptr points to an address
    // that is aligned.
    void store_a(ValueType* val_ptr) const;
    // Member function storing to aligned uncached memory (non-temporal store).
    // This may be more efficient than store_a when storing large blocks of memory if it
    // is unlikely that the data will stay in the cache until it is read again.
    // Note: Will generate runtime error if p is not aligned
    void store_stream(ValueType* val_ptr) const;
    // Partial store. Store n elements
    void store_partial(int n, ValueType* val_ptr) const;

    // ------------------------------------------------------
    // Gather and Scatter
    // ------------------------------------------------------
    SimdClassBase& gather(const ValueType* val_ptr, const SimdClassBase<int, tag>& indices);
    void scatter(ValueType* val_ptr, const SimdClassBase<int, tag>& indices) const;

    // ------------------------------------------------------
    // Data member accessors
    // ------------------------------------------------------
    simd_type& simd()
    {
        return data_.simd_;
    }
    const simd_type& simd() const
    {
        return data_.simd_;
    }
    // Extract a single element. Use store function if extracting more than one element.
    // Operator [] can only read an element, not write.
    ValueType operator[](int index) const;

    // ------------------------------------------------------
    // Data elements manipulation
    // ------------------------------------------------------
    // Member function to change a single element in vector
    SimdClassBase& insert(size_t index, ValueType val);
    // Member function to change a single element in copy of vector which is returned
    SimdClassBase insertCopy(size_t index, ValueType val) const;
    // Cut off vector to n elements. The last (SimdLen - n) elements are set to zero
    SimdClassBase& cutoff(size_t n);
    // Cut off copy of vector to n elements. The last (SimdLen - n) elements are set to zero
    SimdClassBase cutoffCopy(size_t n) const;
    // Shift left elements of vector by n. The empty lanes are filled with zero.
    SimdClassBase& shiftLeft(size_t n);
    // Shift left elements of copy of vector by n. The empty lanes are filled with zero.
    SimdClassBase shiftLeftCopy(size_t n) const;
    // Shift right elements of vector by n. The empty lanes are filled with zero.
    SimdClassBase& shiftRight(size_t n);
    // Shift right elements of copy of vector by n. The empty lanes are filled with zero.
    SimdClassBase shiftRightCopy(size_t n) const;
    // Cyclically rotate/shift elements of vector by n.
    SimdClassBase& rotate(size_t n);
    // Cyclically rotate/shift elements of copy of vector by n.
    SimdClassBase rotateCopy(size_t n) const;
    // Get copy of vector with only sign bits set for each lane.
    SimdClassBase sign() const
    {
        return SimdClassBase{ Detail::sign<simd_type>(data_.simd_) };
    }

    // ------------------------------------------------------
    // Print I/O
    // ------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& stream,
                                    const SimdClassBase& a)
    {
        Detail::print<simd_type>(stream, a.data_.simd_);
        return stream;
    }

    // ------------------------------------------------------
    // Basic Arithmetic
    // ------------------------------------------------------
    friend SimdClassBase operator-(const SimdClassBase& a)
    {
        return SimdClassBase{ Detail::minus<simd_type>(a.data_.simd_) };
    }
    friend SimdClassBase operator+(const SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        return SimdClassBase{ Detail::add<simd_type>(a.data_.simd_,
                                                     b.data_.simd_) };
    }
    friend SimdClassBase& operator+=(SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        a = a + b;
        return a;
    }
    friend SimdClassBase operator-(const SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        return SimdClassBase{ Detail::substract<simd_type>(a.data_.simd_,
                                                           b.data_.simd_) };
    }
    friend SimdClassBase& operator-=(SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        a = a - b;
        return a;
    }
    friend SimdClassBase operator*(const SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        return SimdClassBase{ Detail::multiply<simd_type>(a.data_.simd_,
                                                          b.data_.simd_) };
    }
    friend SimdClassBase& operator*=(SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        a = a * b;
        return a;
    }
    friend SimdClassBase operator/(const SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        return SimdClassBase{ Detail::divide<simd_type>(a.data_.simd_,
                                                        b.data_.simd_) };
    }
    friend SimdClassBase& operator/=(SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        a = a / b;
        return a;
    }

    friend SimdClassBase min(const SimdClassBase& a, const SimdClassBase& b)
    {
        return SimdClassBase{ Detail::min<simd_type>(a.data_.simd_,
                                                     b.data_.simd_) };
    }
    friend SimdClassBase max(const SimdClassBase& a, const SimdClassBase& b)
    {
        return SimdClassBase{ Detail::max<simd_type>(a.data_.simd_,
                                                     b.data_.simd_) };
    }

    //
    ValueType min() const
    {
        return ValueType{ Detail::min<value_type, simd_type>(data_.simd_) };
    }
    ValueType max() const
    {
        return ValueType{ Detail::max<value_type, simd_type>(data_.simd_) };
    }

    friend SimdClassBase sqrt(const SimdClassBase& a)
    {
        return SimdClassBase{ Detail::sqrt<simd_type>(a.data_.simd_) };
    }
    /* Reciprocal( inverse) Square Root */
    friend SimdClassBase rsqrt(const SimdClassBase& a)
    {
        return SimdClassBase{ Detail::rsqrt<simd_type>(a.data_.simd_) };
    }
    friend SimdClassBase abs(const SimdClassBase& a)
    {
        return SimdClassBase{ Detail::abs<simd_type>(a.data_.simd_) };
    }
    friend SimdClassBase log(const SimdClassBase& a)
    {
        return SimdClassBase{ Detail::log<simd_type>(a.data_.simd_) };
    }
    friend SimdClassBase pow(const SimdClassBase& a, int exp)
    {
        return SimdClassBase{ Detail::pow<simd_type>(a.data_.simd_, exp) };
    }

    /* Logical */
    template <typename T = void,
              typename std::enable_if<std::is_same<int, ValueType>::value, T>::type* = nullptr>
    friend SimdClassBase operator<<(const SimdClassBase& a,
                                   int b)
    {
        return SimdClassBase{ Detail::shiftLBits<simd_type>(a.data_.simd_,
                                                        b) };
    }
    template <typename T = void,
              typename std::enable_if<std::is_same<int, ValueType>::value, T>::type* = nullptr>
    friend SimdClassBase operator>>(const SimdClassBase& a,
                                   int b)
    {
        return SimdClassBase{ Detail::shiftRBits<simd_type>(a.data_.simd_,
                                                        b) };
    }
    friend SimdClassBase operator&(const SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        return SimdClassBase{ Detail::ANDBits<simd_type>(a.data_.simd_,
                                                        b.data_.simd_) };
    }
    friend SimdClassBase operator|(const SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        return SimdClassBase{ Detail::ORBits<simd_type>(a.data_.simd_,
                                                        b.data_.simd_) };
    }
    friend SimdClassBase operator^(const SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        return SimdClassBase{ Detail::XORBits<simd_type>(a.data_.simd_,
                                                        b.data_.simd_) };
    }

    // Comparison (mask returned)
    friend SimdMaskBase<tag> operator<(const SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        return SimdMaskBase<tag>{Detail::lessThan<simd_type>(a.data_.simd_,
                                                            b.data_.simd_)};
    }
    friend SimdMaskBase<tag> operator<=(const SimdClassBase& a,
                                    const SimdClassBase& b)
    {
        return SimdMaskBase<tag>{Detail::lessThanEqual<simd_type>(
            a.data_.simd_, b.data_.simd_)};
    }
    friend SimdMaskBase<tag> operator>(const SimdClassBase& a,
                                   const SimdClassBase& b)
    {
        return SimdMaskBase<tag>{Detail::greaterThan<simd_type>(a.data_.simd_,
                                                               b.data_.simd_)};
    }
    friend SimdMaskBase<tag> operator>=(const SimdClassBase& a,
                                    const SimdClassBase& b)
    {
        return SimdMaskBase<tag>{Detail::greaterThanEqual<simd_type>(
            a.data_.simd_, b.data_.simd_)};
    }
    friend SimdMaskBase<tag> operator==(const SimdClassBase& a,
                                    const SimdClassBase& b)
    {
        return SimdMaskBase<tag>{Detail::equal<simd_type>(a.data_.simd_,
                                                         b.data_.simd_)};
    }
    friend SimdMaskBase<tag> operator!=(const SimdClassBase& a,
                                    const SimdClassBase& b)
    {
        return SimdMaskBase<tag>{Detail::notEqual<simd_type>(a.data_.simd_,
                                                            b.data_.simd_)};
    }
    friend SimdMaskBase<tag> operator<(const SimdClassBase& a, ValueType val)
    { // mask returned
        return (a < SimdClassBase{ val });
    }
    friend SimdMaskBase<tag> operator<=(const SimdClassBase& a, ValueType val)
    { // mask returned
        return (a <= SimdClassBase{ val });
    }
    friend SimdMaskBase<tag> operator>(const SimdClassBase& a, ValueType val)
    { // mask returned
        return (a > SimdClassBase{ val });
    }
    friend SimdMaskBase<tag> operator>=(const SimdClassBase& a, ValueType val)
    { // mask returned
        return (a >= SimdClassBase{ val });
    }
    friend SimdMaskBase<tag> operator==(const SimdClassBase& a, ValueType val)
    { // mask returned
        return (a == SimdClassBase{ val });
    }
    friend SimdMaskBase<tag> operator!=(const SimdClassBase& a, ValueType val)
    { // mask returned
        return (a != SimdClassBase{ val });
    }
    friend SimdMaskBase<tag> operator<(ValueType val, const SimdClassBase& a)
    { // mask returned
        return (SimdClassBase{ val } < a);
    }
    friend SimdMaskBase<tag> operator<=(ValueType val, const SimdClassBase& a)
    { // mask returned
        return (SimdClassBase{ val } <= a);
    }
    friend SimdMaskBase<tag> operator>(ValueType val, const SimdClassBase& a)
    { // mask returned
        return (SimdClassBase{ val } > a);
    }
    friend SimdMaskBase<tag> operator>=(ValueType val, const SimdClassBase& a)
    { // mask returned
        return (SimdClassBase{ val } >= a);
    }
    friend SimdMaskBase<tag> operator==(ValueType val, const SimdClassBase& a)
    { // mask returned
        return (SimdClassBase{ val } == a);
    }
    friend SimdMaskBase<tag> operator!=(ValueType val, const SimdClassBase& a)
    { // mask returned
        return (SimdClassBase{ val } != a);
    }

protected:
    SimdData<ValueType, tag> data_;
};

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_CLASS_H
