// -*- C++ -*-
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../simd.h"
#include <iostream>

using KFP::SIMD::simd_mask;
using KFP::SIMD::simd_index;
using KFP::SIMD::simd_float;

TEST_CASE("Testing simd_mask") {
    simd_mask mask;
    SUBCASE("Testing default state") {
        CHECK(mask == simd_mask{false});
        CHECK(mask.AND() == false);
        CHECK(mask.OR() == false);
        CHECK(mask.count() == 0);
    }

    simd_mask mask_true{true};
    SUBCASE("Testing constant constructor") {
        CHECK(mask_true.AND() == true);
        CHECK(mask_true.OR() == true);
        CHECK(mask_true.count() == 4);
    }
}
TEST_CASE("Testing methods") {
    const simd_float val_const = simd_float(1.0f) ;
    const simd_float val_10 = simd_float(10.0f) ;
    SUBCASE("Testing the single value constructor") {
        CHECK((val_const == simd_float(1.0f)).AND() );
    }
    SUBCASE("Testing insert") {
        const simd_float val_zero = simd_float{0.0f} ;
        const simd_float val_1 =  simd_float(1.0f);
        const simd_float val_2 =  simd_float(2.0f);
        const simd_float val_3 =  simd_float(3.0f);
        const simd_float val_4 =  simd_float(4.0f);
        CHECK((simd_float{0.0f}.insert(0, 1.0f) == val_1).AND()) ;
        CHECK((simd_float{0.0f}.insert(1, 2.0f) == val_2).AND());
        CHECK((simd_float{0.0f}.insert(2, 3.0f) == val_3).AND());
        CHECK((simd_float{0.0f}.insert(3, 4.0f) == val_4).AND());
        SUBCASE("Testing addition version 1") {
            CHECK((val_10 == (val_1 + val_2 + val_3 + val_1)).count() == 2);
            CHECK((val_10 == (val_1 + val_2 + val_3 + val_4)).AND());
        }
    }
}
