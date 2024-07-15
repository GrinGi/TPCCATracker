// -*- C++ Header -*-
/*
==================================================
Authors: A.Mithran, P. Kisel, I. Kisel
Emails: mithran@fias.uni-frankfurt.de
==================================================
*/

#ifndef SIMD_ALLOCATE_H
#define SIMD_ALLOCATE_H

#include <cstdint>
#include <cassert>
#include <limits>
#include <vector>

namespace KFP
{
namespace SIMD
{

constexpr bool isAlignment(std::size_t N)
{
    return (N > 0) && ((N & (N - 1)) == 0);
}

inline void*
alignedAllocate(std::size_t size, std::size_t alignment)
{
    assert((isAlignment(alignment) && "[Error] (alignedAllocate): Invalid value given for aligment"));
    if(!size) return nullptr;
    constexpr std::size_t voidptr_Alignment = alignof(void*);
    if (alignment < voidptr_Alignment) {
        alignment = voidptr_Alignment;
    }
    constexpr std::size_t voidptr_Size = sizeof(void*);
    std::size_t buffer_size = size + alignment + voidptr_Size;
    void* aligned{nullptr};
    void* original = ::operator new(buffer_size);
    if (original) {
        aligned = static_cast<void*>(static_cast<char*>(original) + voidptr_Size);
        // aligned = static_cast<void*>(static_cast<uint8_t*>(original) + voidptr_Size);
        buffer_size -= voidptr_Size;
        const auto tmp = reinterpret_cast<uintptr_t>(aligned) + alignment - 1;
        aligned = reinterpret_cast<void*>(tmp & ~(alignment-1));
        *(static_cast<void**>(aligned) - 1) = original;
        return aligned;
        // if (std::align(alignment, size, aligned, buffer_size)){
        //     *(static_cast<void**>(aligned) - 1) = original;
        //     return aligned;
        // }
    }
    return nullptr;
}

inline void
alignedDeallocate(void* ptr)
{
    if (ptr) {
        ::operator delete(*(static_cast<void**>(ptr) - 1));
    }
}

#define SETUP_ALIGNED_OPERATOR_NEW_DELETE(Alignment)                                                                    \
    void* operator new(std::size_t size)                                                                                \
    {                                                                                                                   \
        return KFP::SIMD::alignedAllocate<Alignment>(size);                                                            \
    }                                                                                                                   \
    void* operator new[](std::size_t size)                                                                              \
    {                                                                                                                   \
        return KFP::SIMD::alignedAllocate<Alignment>(size);                                                            \
    }                                                                                                                   \
    void operator delete(void* ptr)                                                                                     \
    {                                                                                                                   \
        KFP::SIMD::alignedDeallocate(ptr);                                                                             \
    }                                                                                                                   \
    void operator delete[](void* ptr)                                                                                   \
    {                                                                                                                   \
        KFP::SIMD::alignedDeallocate(ptr);                                                                             \
    }                                                                                                                   \
    void operator delete(void* ptr, std::size_t /* sz */)                                                               \
    {                                                                                                                   \
        KFP::SIMD::alignedDeallocate(ptr);                                                                             \
    }                                                                                                                   \
    void operator delete[](void* ptr, std::size_t /* sz */)                                                             \
    {                                                                                                                   \
        KFP::SIMD::alignedDeallocate(ptr);                                                                             \
    }                                                                                                                   \
    void* operator new(std::size_t size, void* ptr) { return ::operator new(size, ptr); }                               \
    void* operator new[](std::size_t size, void* ptr) { return ::operator new[](size, ptr); }                           \
    void operator delete(void* memory, void* ptr)                                                                       \
    {                                                                                                                   \
        return ::operator delete(memory, ptr);                                                                          \
    }                                                                                                                   \
    void operator delete[](void* memory, void* ptr)                                                                     \
    {                                                                                                                   \
        return ::operator delete[](memory, ptr);                                                                        \
    }                                                                                                                   \


template<typename T, std::size_t Alignment>
class AlignedAllocator {
    static_assert(isAlignment(Alignment), "[Error] (AlignedAllocator): Invalid value given for aligment");
    static_assert(
        Alignment >= alignof(T),
        "[Error] (AlignedAllocator): Types like int have minimum alignment requirements or access will result in crashes."
    );
    // assert((isAlignment(Alignment) && "[Error] (AlignedAllocator): Invalid value given for aligment"));
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef void* void_pointer;
    typedef const void* const_void_pointer;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template<typename U>
    struct rebind {
        typedef AlignedAllocator<U, Alignment> other;
    };

    AlignedAllocator() noexcept
    {}
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept
    {}

    pointer address(reference x) const noexcept
    { return std::addressof(x); }

    const_pointer address(const_reference x) const noexcept
    { return std::addressof(x); }

    pointer allocate(size_type size, const_void_pointer = nullptr)
    {
        if (size == 0) {
            return nullptr;
        }
        void* p = alignedAllocate(sizeof(T) * size, Alignment);
        if (!p) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(p);
    }

    void deallocate(pointer ptr, size_type) noexcept
    {
        alignedDeallocate(ptr);
    }

    size_type max_size() const noexcept
    {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }

    template<typename U, class V>
    void construct(U* ptr, const V& value) {
        ::new(static_cast<void*>(ptr)) U(value);
    }

    template<typename U>
    void construct(U* ptr) {
        ::new(static_cast<void*>(ptr)) U();
    }

    template<typename U>
    void destroy(U* ptr) {
        ptr->~U();
    }
};

template<typename T, typename U, std::size_t Alignment>
inline bool
operator==(const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept
{
    return true;
}

template<typename T, typename U, std::size_t Alignment>
inline bool
operator!=(const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept
{
    return false;
}

template<typename T, std::size_t Alignment = alignof(T)>
using Vector = std::vector<T, AlignedAllocator<T, Alignment>>;

} // namespace SIMD
} // namespace KFP

#endif
