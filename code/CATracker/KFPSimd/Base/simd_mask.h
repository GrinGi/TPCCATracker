// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran;
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_MASK_H
#define SIMD_MASK_H

#include "simd_tag.h"
#include "simd_data.h"
#include "simd_detail.h"

#include <iostream>

namespace KFP {
namespace SIMD {

template <Tag tag> class SimdMaskBase
{
public:
    typedef typename SimdData<int, tag>::simd_type simd_typei;
    typedef typename SimdData<float, tag>::simd_type simd_typef;
    typedef typename SimdData<int, tag>::value_type value_typei;
    typedef typename SimdData<float, tag>::value_type value_typef;
    static constexpr Tag tag_val{ tag };
    static constexpr size_t SimdSize{ sizeof(simd_typei) };
    static constexpr size_t SimdLen{ SimdSize / sizeof(int) };

    // ------------------------------------------------------
    // Constructors
    // ------------------------------------------------------
    // Default constructor:
    SimdMaskBase();
    SimdMaskBase(bool val);
    SimdMaskBase(const bool* val_ptr);
    SimdMaskBase(const simd_typei& mask);
    SimdMaskBase(const simd_typef& mask);
    SimdMaskBase(const SimdMaskBase& class_mask);

    // Assignment operators
    SimdMaskBase& operator=(bool val);
    SimdMaskBase& operator=(const simd_typei& mask);
    SimdMaskBase& operator=(const simd_typef& mask);
    SimdMaskBase& operator=(const SimdMaskBase& class_mask);

    // ------------------------------------------------------
    // Load and Store
    // ------------------------------------------------------
    // Member function to load from array.
    SimdMaskBase& load(const bool* val_ptr);
    // Member function to store into array.
    void store(bool* val_ptr) const;

    // ------------------------------------------------------
    // Status accessors
    // ------------------------------------------------------
    int count() const;
    bool AND() const;
    bool OR() const;

    // ------------------------------------------------------
    // Data member accessors
    // ------------------------------------------------------
    simd_typei& maski()
    {
        return mask_;
    }
    const simd_typei& maski() const
    {
        return mask_;
    }
    simd_typef maskf() const;
    bool operator[](int index) const;

    // ------------------------------------------------------
    // Data elements manipulation
    // ------------------------------------------------------
    // Member function to change a single element in vector
    SimdMaskBase& insert(size_t index, bool val);
    // Member function to change a single element in copy of vector which is returned
    SimdMaskBase insertCopy(size_t index, bool val) const;
    // Cut off vector to n elements. The last (SimdLen - n) elements are set to false
    SimdMaskBase& cutoff(size_t index);
    // Cut off copy of vector to n elements. The last (SimdLen - n) elements are set to false
    SimdMaskBase cutoffCopy(size_t index) const;

    // ------------------------------------------------------
    // Print I/O
    // ------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& stream, const SimdMaskBase& a)
    {
        Detail::print<SimdMaskBase>(stream, a);
        return stream;
    }

    // Comparison (mask returned)
    friend SimdMaskBase operator!(const SimdMaskBase& a)
    {
        return SimdMaskBase{ Detail::NOTLanes<simd_typei>(a.mask_) };
    }
    friend bool operator==(const SimdMaskBase& a, const SimdMaskBase& b)
    {
        return Detail::equal<bool, simd_typei, simd_typei>(a.mask_, b.mask_);
    }
    friend bool operator!=(const SimdMaskBase& a, const SimdMaskBase& b)
    {
        return Detail::notEqual<bool, simd_typei, simd_typei>(a.mask_, b.mask_);
    }
    friend SimdMaskBase operator^(const SimdMaskBase& a, const SimdMaskBase& b)
    {
        return SimdMaskBase{ Detail::XORLanes<simd_typei>(a.mask_, b.mask_) };
    }
    friend SimdMaskBase operator&&(const SimdMaskBase& a, const SimdMaskBase& b)
    {
        return SimdMaskBase{ Detail::ANDLanes<simd_typei>(a.mask_, b.mask_) };
    }
    friend SimdMaskBase operator||(const SimdMaskBase& a, const SimdMaskBase& b)
    {
        return SimdMaskBase{ Detail::ORLanes<simd_typei>(a.mask_, b.mask_) };
    }
    friend SimdMaskBase& operator&=(SimdMaskBase& a, const SimdMaskBase& b)
    {
        a = a && b;
        return a;
    }
    friend SimdMaskBase& operator|=(SimdMaskBase& a, const SimdMaskBase& b)
    {
        a = a || b;
        return a;
    }

    ///
    bool isEmpty() const;
    bool isFull() const;

protected:
    simd_typei mask_;
};

} // namespace SIMD
} // namespace KFP

#endif // !SIMD_MASK_H
