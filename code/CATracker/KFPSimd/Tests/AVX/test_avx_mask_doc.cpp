// -*- C++ -*-
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../simd.h"
#include <iostream>

using KFP::SIMD::simd_mask;

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

    bool masks[]{true, false, false, false};
    simd_mask mask_1true3false{masks};
    SUBCASE("Testing variable constructor") {
        CHECK(mask_1true3false.AND() == false);
        CHECK(mask_1true3false.OR() == true);
        CHECK(mask_1true3false.count() == 1);
    }
}
