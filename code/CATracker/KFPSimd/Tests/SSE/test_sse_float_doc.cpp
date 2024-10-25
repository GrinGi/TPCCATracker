// -*- C++ -*-
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../doctest.h"

#include "../../simd.h"
#include <iostream>

using KFP::SIMD::simd_float;

TEST_CASE("Testing methods") {
    const float fone[]{1.0f, 1.0f, 1.0f, 1.0f};
    const float f1234[]{1.0f, 2.0f, 3.0f, 4.0f};
    const float f2341[]{2.0f, 3.0f, 4.0f, 1.0f};
    const float f3412[]{3.0f, 4.0f, 1.0f, 2.0f};
    const float f4123[]{4.0f, 1.0f, 2.0f, 3.0f};
    const float f4321[]{4.0f, 3.0f, 2.0f, 1.0f};
    [[maybe_unused]] const simd_float val_zero = simd_float{0.0f} ;
    [[maybe_unused]] const simd_float val_one = simd_float(fone) ;
    [[maybe_unused]] const simd_float val_1234 = simd_float(f1234) ;
    [[maybe_unused]] const simd_float val_2341 = simd_float(f2341) ;
    [[maybe_unused]] const simd_float val_3412 = simd_float(f3412) ;
    [[maybe_unused]] const simd_float val_4123 = simd_float(f4123) ;
    [[maybe_unused]] const simd_float val_4321 = simd_float(f4321) ;
    SUBCASE("Testing the single value constructor") {
        CHECK((val_one == simd_float(1.0f)).AND() );
    }
    SUBCASE("Testing insert") {
        const float f1000[]{1.0f, 0.0f, 0.0f, 0.0f};
        const float f0200[]{0.0f, 2.0f, 0.0f, 0.0f};
        const float f0030[]{0.0f, 0.0f, 3.0f, 0.0f};
        const float f0004[]{0.0f, 0.0f, 0.0f, 4.0f};
        const simd_float val_1000 = simd_float(f1000);
        const simd_float val_0200 = simd_float(f0200);
        const simd_float val_0030 = simd_float(f0030);
        const simd_float val_0004 = simd_float(f0004);
        CHECK((simd_float{0.0f}.insert(0, 1.0f) == val_1000).AND()) ;
        CHECK((simd_float{0.0f}.insert(1, 2.0f) == val_0200).AND());
        CHECK((simd_float{0.0f}.insert(2, 3.0f) == val_0030).AND());
        CHECK((simd_float{0.0f}.insert(3, 4.0f) == val_0004).AND());
        SUBCASE("Testing addition version") {
            CHECK((val_1234 == (val_1000 + val_0200 + val_0030 + val_1000)).count() == 2);
            CHECK((val_1234 == (val_1000 + val_0200 + val_0030 + val_0004)).AND());
        }
        SUBCASE("Testing insert, shift, rotate") {
            CHECK((val_zero.insertCopy(0, 1) == (val_1000)).count() == 4);
            std::cout << "val_zero.insertCopy(2, 2) : " << val_zero.insertCopy(2, 2) << '\n';
            std::cout << "(val_0200).shiftRightCopy(1) : " << (val_0200).shiftRightCopy(1) << '\n';
            CHECK((val_zero.insertCopy(2, 2) == (val_0200).shiftRightCopy(1)).count() == 4);
            std::cout << "val_zero.insertCopy(0, 1)*2 : " << val_zero.insertCopy(0, 1)*2 << '\n';
            std::cout << "(val_0200).shiftLeftCopy(1) : " << (val_0200).shiftLeftCopy(1) << '\n';
            CHECK((val_zero.insertCopy(0, 1)*2 == (val_0200).shiftLeftCopy(1)).count() == 4);
            std::cout << "val_3412 : " << val_3412 << '\n';
            std::cout << "(val_1234).rotateCopy(2) : " << (val_1234).rotateCopy(2) << '\n';
            CHECK((val_3412 == (val_1234).rotateCopy(2)).count() == 4);
            std::cout << "val_2341 : " << val_2341 << '\n';
            std::cout << "(val_1234).rotateCopy(1) : " << (val_1234).rotateCopy(1) << '\n';
            CHECK((val_2341 == (val_1234).rotateCopy(1)).count() == 4);
        }
    }
}
