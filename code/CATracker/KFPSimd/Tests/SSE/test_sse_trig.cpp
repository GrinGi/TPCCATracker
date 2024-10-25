// -*- C++ -*-
#include "../../simd.h"
#include <iostream>
#include <Vc/Vc>

using KFP::SIMD::simd_mask;
using KFP::SIMD::simd_float;
using KFP::SIMD::simd_int;
using Vc::float_m;
using Vc::float_v;
using Vc::int_v;
using Vc::uint_v;

__KFP_SIMD__INLINE simd_float Sin(const simd_float& phi)
{
    const simd_float pi{3.1415926535897932f};
    const simd_float nTurnsF = (phi + pi) / (simd_float{2.f}*pi);
    simd_int nTurns = simd_int{nTurnsF};
    nTurns = select((nTurns <= 0) && (phi < -pi), nTurns-1, nTurns);

    const simd_float& x{ phi - simd_float(nTurns)*(simd_float(2.f)*pi) };

    const simd_float& B = simd_float{4.f}/pi;
    const simd_float& C = -B/pi;

    simd_float y = (B + C * abs(x)) * x;

    const simd_float& P{ 0.218f };
    y = P * (y * abs(y) - y) + y;

    return y;
}
__KFP_SIMD__INLINE simd_float Cos(const simd_float& phi)
{
    return Sin(phi + simd_float{1.570796326795f}); //x + pi/2
}

static inline __attribute__((always_inline)) simd_float multiplySign(const simd_int& s, const simd_float& x) {
    const simd_float& sign_val = simd_float::type_cast(s.sign());
    return simd_float{ sign_val ^ x };
}

static __KFP_SIMD__INLINE simd_float sinSeries(const simd_float x) {
    const simd_float x2 = x*x;
    simd_float y = -1.984126984e-4f;
    y = y * x2 + 8.333333333e-3f;
    y = y * x2 - 1.666666667e-1f;
    return y * (x2 * x) + x;
}

static __KFP_SIMD__INLINE simd_float cosSeries(const simd_float x) {
    const simd_float x2 = x*x;
    simd_float y = 2.48015873e-5f;
    y = y * x2 - 1.388888889e-03f;
    y = y * x2 + 4.166666667e-2f;
    return y * (x2 * x2) - .5f * x2 + 1.f;
}

static __KFP_SIMD__INLINE void sincos(simd_float x, simd_float& sinX, simd_float& cosX) {
#if 1
    //     const simd_float sin0 = sin(x);
    //     const simd_float cos0 = cos(x);

    const simd_float pi2i = 6.36619772e-1f;
    const simd_int nPi2 = static_cast<simd_int>( round(x*pi2i) );
    const simd_int q = nPi2 & 3;
    const simd_int sinSign = q << 30;
    const simd_int cosSign = (q+1) << 30;

    const simd_float nPi2f = static_cast<simd_float>(nPi2);
    x = x - simd_float{1.5707969666f} * nPi2f;
    x = x + simd_float{6.3975784e-7f} * nPi2f;

    const simd_float sinS = sinSeries(x);
    const simd_float cosS = cosSeries(x);
    // std::cout << "sinS : " << sinS << '\n';
    // std::cout << "cosS : " << cosS << '\n';

    const simd_mask mask = (q == simd_int{0} || q == simd_int{2});
    sinX = multiplySign( sinSign, select(mask, sinS, cosS) );
    cosX = multiplySign( cosSign, select(mask, cosS, sinS) );
    // sinX = select(mask, sinS, cosS);
    // sinX = multiplySign(sinSign, sinX);
    // cosX = select(mask, cosS, sinS);
    // cosX = multiplySign(cosSign, cosX);

    //     for(int i=0; i<4; i++) {
    //       if(fabs(sinX[i] - sin0[i]) > 1.e-7f * fabs(sin0[i]) || fabs(cosX[i] - cos0[i]) > 1.e-7f * fabs(cos0[i]) ) {
    //         std::cout << "x " << x << "   npi2 " << nPi2 << std::endl;
    //         std::cout << "sin " << sinX << "     " << sin0 << " " << sin(x) << std::endl;
    //         std::cout << "cos " << cosX << "     " << cos0 << " " << cos(x) << std::endl;
    //         std::cin.get();
    //       }
    //     }
#else
    const simd_float sin0 = sin(x);
    const simd_float cos0 = cos(x);

    constexpr double pi2 = 1.5707963267948966192313216916398;
    constexpr double pi2i = 0.63661977236758134307553505349006;

    typedef Vc::SimdArray<double, 4> TD;
    TD xd = simd_cast<TD>(x);
    TD nPi2 = round(xd * pi2i);
    x = simd_cast<simd_float>(xd - nPi2 * pi2);
    const simd_int q = simd_cast<simd_int>(nPi2) & 3;

    const simd_int sinSign = q << 30;
    const simd_int cosSign = (q+1) << 30;

    const simd_float sinS = sinSeries(x);
    const simd_float cosS = cosSeries(x);

    const float_m mask = simd_cast<float_m>(q == 0 || q == 2);
    sinX = multiplySign( sinSign, iif(mask, sinS, cosS) );
    cosX = multiplySign( cosSign, iif(mask, cosS, sinS) );

    for(int i=0; i<4; i++) {
      if(fabs(sinX[i] - sin0[i]) > 1.e-7f * fabs(sin0[i]) || fabs(cosX[i] - cos0[i]) > 1.e-7f * fabs(cos0[i]) ) {
        std::cout << "x " << x << "   npi2 " << nPi2 << std::endl;
        std::cout << "sin " << sinX << "     " << sin0 << " " << sin(x) << std::endl;
        std::cout << "cos " << cosX << "     " << cos0 << " " << cos(x) << std::endl;
        std::cin.get();
      }
    }
#endif
}

static __KFP_SIMD__INLINE simd_float ATan2( const simd_float &y, const simd_float &x )
{
    const simd_float pi(3.1415926535897932f);
    const simd_float zero(0.0f);

    const simd_mask& xZero = (x == zero);
    const simd_mask& yZero = (y == zero);
    const simd_mask& xNeg  = (x < zero);
    const simd_mask& yNeg  = (y < zero);

    const simd_float& absX = abs(x);
    const simd_float& absY = abs(y);

    simd_float a = absY / absX;
    const simd_mask& gt_tan_3pi_8 = (a > simd_float(2.414213562373095f));
    const simd_mask& gt_tan_pi_8  = (a > simd_float(0.4142135623730950f)) && (!gt_tan_3pi_8);
    simd_float b{0.0f};
    b = KFP::SIMD::select(gt_tan_3pi_8, simd_float{pi/2.f}, b);
    b = KFP::SIMD::select(gt_tan_pi_8, simd_float{pi/4.f}, b);
    a = KFP::SIMD::select(gt_tan_3pi_8, simd_float{-1.f / a}, a);
    a = KFP::SIMD::select(gt_tan_pi_8, simd_float((absY - absX) / (absY + absX)), a) ;
    const simd_float& a2 = a * a;
    b = b + (((8.05374449538e-2f * a2
          - 1.38776856032E-1f) * a2
          + 1.99777106478E-1f) * a2
          - 3.33329491539E-1f) * a2 * a
          + a;
    b = KFP::SIMD::select((xNeg ^ yNeg), -b, b);
    b = KFP::SIMD::select((xNeg && !yNeg), (b+pi), b);
    b = KFP::SIMD::select((xNeg &&  yNeg), (b-pi), b);
    b = KFP::SIMD::select((xZero && yZero), zero, b);
    b = KFP::SIMD::select((xZero &&  yNeg), (-pi/2.f), b);
    return b;
}

__m128 asin_sse_better_approx1(__m128 x) {
    // Constants from the Taylor series expansion of asin(x)
    const __m128 one_third = _mm_set1_ps(1.0f / 3.0f);
    const __m128 one_fifth = _mm_set1_ps(1.0f / 5.0f);
    const __m128 one_seventh = _mm_set1_ps(1.0f / 7.0f);
    const __m128 one_ninth = _mm_set1_ps(1.0f / 9.0f);

    // Coefficients for higher order terms, adjusted for simplicity and demonstration
    const __m128 c3 = _mm_set1_ps(0.166666666f); // (1/2)*1/3 for x^3 term
    const __m128 c5 = _mm_set1_ps(0.075f);       // (1*3)/(2*4)*1/5 for x^5 term
    const __m128 c7 = _mm_set1_ps(0.0446428571f); // (1*3*5)/(2*4*6)*1/7 for x^7 term
    const __m128 c9 = _mm_set1_ps(0.0303819444f); // (1*3*5*7)/(2*4*6*8)*1/9 for x^9 term

    __m128 x2 = _mm_mul_ps(x, x); // x^2
    __m128 x3 = _mm_mul_ps(x2, x); // x^3
    __m128 x5 = _mm_mul_ps(x3, x2); // x^5
    __m128 x7 = _mm_mul_ps(x5, x2); // x^7
    __m128 x9 = _mm_mul_ps(x7, x2); // x^9

    // Compute the terms of the polynomial approximation
    __m128 term3 = _mm_mul_ps(c3, x3);
    __m128 term5 = _mm_mul_ps(c5, x5);
    __m128 term7 = _mm_mul_ps(c7, x7);
    __m128 term9 = _mm_mul_ps(c9, x9);

    // Combine the terms
    __m128 result = x; // Start with the linear term
    result = _mm_add_ps(result, term3);
    result = _mm_add_ps(result, term5);
    result = _mm_add_ps(result, term7);
    result = _mm_add_ps(result, term9);

    return result;
}

__m128 asin_sse_better_approx2(__m128 x) {
    // Example coefficients for a conceptual polynomial approximation.
    // In practice, you should use coefficients derived from an accurate source.
    const __m128 c1 = _mm_set1_ps(1.0f);
    const __m128 c3 = _mm_set1_ps(0.16666666666f); // 1/6 for x^3 term
    const __m128 c5 = _mm_set1_ps(0.075f);         // Approximation for x^5 term
    const __m128 c7 = _mm_set1_ps(0.04464285714f); // Approximation for x^7 term

    __m128 x2 = _mm_mul_ps(x, x);  // x^2
    __m128 x3 = _mm_mul_ps(x2, x); // x^3
    __m128 x5 = _mm_mul_ps(x2, x3); // x^5
    __m128 x7 = _mm_mul_ps(x2, x5); // x^7

    // Compute the polynomial approximation: x + c3*x^3 + c5*x^5 + c7*x^7
    __m128 result = _mm_add_ps(x, _mm_mul_ps(c3, x3)); // x + c3*x^3
    result = _mm_add_ps(result, _mm_mul_ps(c5, x5));   // + c5*x^5
    result = _mm_add_ps(result, _mm_mul_ps(c7, x7));   // + c7*x^7

    return result;
}

__m128 asin_sse_better_approx3(__m128 x) {
    // Assuming x is in the range [-1, 1]
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 half = _mm_set1_ps(0.5f);

    // Coefficients for a polynomial approximation (these are fabricated for demonstration
    // and do not represent an optimized minimax polynomial for asin)
    const __m128 a = _mm_set1_ps(1.57079632679f); // π/2 for large values close to 1 or -1
    const __m128 b = _mm_set1_ps(0.0f); // Placeholder coefficient
    const __m128 c = _mm_set1_ps(-0.2145988016f); // Example coefficients for small values
    const __m128 d = _mm_set1_ps(0.0886901108f);
    const __m128 e = _mm_set1_ps(-0.0484999454f);

    __m128 abs_x = _mm_andnot_ps(_mm_set1_ps(-0.f), x); // abs(x)
    __m128 x2 = _mm_mul_ps(x, x);
    __m128 x3 = _mm_mul_ps(x2, x);

    // Use a different approximation depending on the range of x
    __m128 result_small = _mm_add_ps(_mm_mul_ps(c, x), _mm_set1_ps(1.57079632679f)); // For abs(x) near 1, simple linear approx
    result_small = _mm_add_ps(result_small, _mm_mul_ps(d, x2)); // Add quadratic term
    result_small = _mm_add_ps(result_small, _mm_mul_ps(e, x3)); // Add cubic term

    __m128 result_large = _mm_sub_ps(a, _mm_sqrt_ps(_mm_sub_ps(one, x))); // For values close to 1 or -1

    // Select which result to use based on the magnitude of x
    __m128 mask_large = _mm_cmpgt_ps(abs_x, half); // abs(x) > 0.5
    __m128 result = _mm_blendv_ps(result_small, result_large, mask_large);

    return result;
}

// static __KFP_SIMD__INLINE simd_float ASin( const simd_float &x )
// {
//     return simd_float{ asin_sse_better_approx1(x.simd()) };
// }

static __KFP_SIMD__INLINE simd_float ASin(simd_float x) {
    // Coefficients for higher order terms based on the polynomial approximation
    const simd_float c3 = 0.166666666f; // For x^3 term
    const simd_float c5 = 0.075f;       // For x^5 term
    const simd_float c7 = 0.0446428571f; // For x^7 term
    const simd_float c9 = 0.0303819444f; // For x^9 term

    simd_mask neg_mask = (x < 0.0f);
    x = abs(x);

    simd_float x2 = (x * x);  // x^2
    simd_float x3 = (x2 * x); // x^3
    simd_float x5 = (x3 * x2); // x^5
    simd_float x7 = (x5 * x2); // x^7
    simd_float x9 = (x7 * x2); // x^9

    // Compute the terms of the polynomial approximation and sum them
    simd_float term3 = (c3 * x3);
    simd_float term5 = (c5 * x5);
    simd_float term7 = (c7 * x7);
    simd_float term9 = (c9 * x9);

    // Combine the terms
    simd_float result = x; // Start with the linear term
    result = (result + term3);
    result = (result + term5);
    result = (result + term7);
    result = (result + term9);

    result = KFP::SIMD::select(neg_mask, -result, result);

    return result;
}

// Extract exponent and mantissa (IEEE 754) and use polynomial approximation for mantissa.
__m128 log_sse1(__m128 x) {
    const __m128 one = _mm_set1_ps(1.0f);

    // Constants for polynomial approximation of ln(x) in the range [1, 2)
    const __m128 a = _mm_set1_ps(0.69314718056f); // ln(2)
    const __m128 b = _mm_set1_ps(2.0f);
    const __m128 c = _mm_set1_ps(-0.33333333333f); // Coefficient for x^2 term (example)
    // More coefficients can be added for higher accuracy

    // Extract exponent and treat mantissa
    __m128i i = _mm_castps_si128(x);
    __m128i exponent = _mm_sub_epi32(_mm_srli_epi32(i, 23), _mm_set1_epi32(127));
    __m128 mantissa = _mm_or_ps(_mm_castsi128_ps(_mm_and_si128(i, _mm_set1_epi32(0x7FFFFF))), one);

    // Compute logarithm for mantissa using polynomial approximation: ln(mantissa) ≈ mantissa + c*mantissa^2 + ...
    __m128 log_mantissa = _mm_add_ps(mantissa, _mm_mul_ps(c, _mm_mul_ps(mantissa, mantissa))); // Simplified
    // Adjust log_mantissa based on actual polynomial used

    // Convert exponent to float, multiply by ln(2), and add to the log of the mantissa
    __m128 log_exponent = _mm_mul_ps(_mm_cvtepi32_ps(exponent), a);
    __m128 result = _mm_add_ps(log_exponent, log_mantissa);

    return result;
}

__m128 log_sse2(__m128 x) {
    // Constants
    const __m128 log2_e = _mm_set1_ps(std::log2(std::exp(1.0f))); // log2(e)
    const __m128 one = _mm_set1_ps(1.0f);

    // Polynomial coefficients for log(mantissa) approximation
    // Example coefficients; for real use, optimize these based on a minimax polynomial or other accurate source
    const __m128 c1 = _mm_set1_ps(0.9999964239f);
    const __m128 c2 = _mm_set1_ps(-0.4998741238f);
    const __m128 c3 = _mm_set1_ps(0.3317990258f);
    const __m128 c4 = _mm_set1_ps(-0.2407338084f);
    const __m128 c5 = _mm_set1_ps(0.1676540711f);

    // Extract exponent and normalize mantissa
    __m128i ix = _mm_castps_si128(x);
    __m128i exp = _mm_sub_epi32(_mm_srli_epi32(ix, 23), _mm_set1_epi32(127));
    __m128 mantissa = _mm_or_ps(_mm_castsi128_ps(_mm_and_si128(ix, _mm_set1_epi32(0x7FFFFF))), one);

    // Normalize mantissa to [0.5, 1.0) range
    mantissa = _mm_mul_ps(mantissa, _mm_set1_ps(0.5f));
    exp = _mm_add_epi32(exp, _mm_set1_epi32(1));

    // Polynomial approximation for log(mantissa) in the range [0.5, 1.0)
    __m128 log_mantissa = _mm_mul_ps(c1, mantissa);
    __m128 m2 = _mm_mul_ps(mantissa, mantissa);
    log_mantissa = _mm_add_ps(log_mantissa, _mm_mul_ps(c2, m2));
    __m128 m3 = _mm_mul_ps(m2, mantissa);
    log_mantissa = _mm_add_ps(log_mantissa, _mm_mul_ps(c3, m3));
    __m128 m4 = _mm_mul_ps(m3, mantissa);
    log_mantissa = _mm_add_ps(log_mantissa, _mm_mul_ps(c4, m4));
    __m128 m5 = _mm_mul_ps(m4, mantissa);
    log_mantissa = _mm_add_ps(log_mantissa, _mm_mul_ps(c5, m5));

    // Adjust log_mantissa based on exponent, convert exponent to float for multiplication
    __m128 e = _mm_cvtepi32_ps(exp);
    __m128 log_exp = _mm_mul_ps(e, log2_e);

    // Combine parts: log(x) = exponent * log(2) + log(mantissa)
    __m128 result = _mm_add_ps(log_exp, log_mantissa);

    return result;
}

static __KFP_SIMD__INLINE simd_float Log( const simd_float &x )
{
    return simd_float{ log_sse2(x.simd()) };
}

Vc::float_v SinVc( const Vc::float_v& phi )
{
    const Vc::float_v pi(3.1415926535897932f);
    const Vc::float_v nTurnsF = (phi + pi) / (Vc::float_v(2.f)*pi);
    Vc::int_v nTurns = simd_cast<Vc::int_v>( nTurnsF );
    nTurns( (nTurns<=Vc::int_v(Vc::Zero)) && simd_cast<Vc::int_m>(phi<-pi)) -= 1;

    const Vc::float_v& x = phi - simd_cast<Vc::float_v>(nTurns)*(Vc::float_v(2.f)*pi);

    const Vc::float_v& B = 4.f/pi;
    const Vc::float_v& C = -B/pi;

    Vc::float_v y = (B + C * Vc::abs(x)) * x;

    const Vc::float_v& P = 0.218f;
    y = P * (y * Vc::abs(y) - y) + y;

    return y;
}

static inline __attribute__((always_inline)) float_v multiplySign(const int_v& sign, const float_v& x) {
    const uint_v& signU = reinterpret_cast<const uint_v&>(sign);
    const uint_v& xU = reinterpret_cast<const uint_v&>(x);
    const uint_v signMask(0x80000000U);
    const uint_v& outU = (signU & signMask) ^ xU;
    const float_v& out = reinterpret_cast<const float_v&>(outU);
    return out;
}

static inline __attribute__((always_inline)) float_v sinSeries(const float_v x) {
    const float_v x2 = x*x;
    float_v y = -1.984126984e-4f;
    y = y * x2 + 8.333333333e-3f;
    y = y * x2 - 1.666666667e-1f;
    return y * (x2 * x) + x;
}

static inline __attribute__((always_inline)) float_v cosSeries(const float_v x) {
    const float_v x2 = x*x;
    float_v y = 2.48015873e-5f;
    y = y * x2 - 1.388888889e-03f;
    y = y * x2 + 4.166666667e-2f;
    return y * (x2 * x2) - .5f * x2 + 1.f;
}

static inline __attribute__((always_inline)) void sincos(float_v x, float_v& sinX, float_v& cosX) {
#if 1
//     const float_v sin0 = sin(x);
//     const float_v cos0 = cos(x);

    const float_v pi2i = 6.36619772e-1f;
    const int_v nPi2 = simd_cast<int_v>( Vc::round(x*pi2i) );
    const int_v q = nPi2 & 3;
    const int_v sinSign = q << 30;
    const int_v cosSign = (q+1) << 30;

    const float_v nPi2f = simd_cast<float_v>(nPi2);
    x -= 1.5707969666f * nPi2f;
    x += 6.3975784e-7f * nPi2f;

    const float_v sinS = sinSeries(x);
    const float_v cosS = cosSeries(x);
    // std::cout << "sinS : " << sinS << '\n';
    // std::cout << "cosS : " << cosS << '\n';

    const float_m mask = simd_cast<float_m>(q == 0 || q == 2);
    sinX = multiplySign( sinSign, iif(mask, sinS, cosS) );
    cosX = multiplySign( cosSign, iif(mask, cosS, sinS) );

//     for(int i=0; i<4; i++) {
//       if(fabs(sinX[i] - sin0[i]) > 1.e-7f * fabs(sin0[i]) || fabs(cosX[i] - cos0[i]) > 1.e-7f * fabs(cos0[i]) ) {
//         std::cout << "x " << x << "   npi2 " << nPi2 << std::endl;
//         std::cout << "sin " << sinX << "     " << sin0 << " " << sin(x) << std::endl;
//         std::cout << "cos " << cosX << "     " << cos0 << " " << cos(x) << std::endl;
//         std::cin.get();
//       }
//     }
#else
    const float_v sin0 = sin(x);
    const float_v cos0 = cos(x);

    constexpr double pi2 = 1.5707963267948966192313216916398;
    constexpr double pi2i = 0.63661977236758134307553505349006;

    typedef Vc::SimdArray<double, 4> TD;
    TD xd = simd_cast<TD>(x);
    TD nPi2 = round(xd * pi2i);
    x = simd_cast<float_v>(xd - nPi2 * pi2);
    const int_v q = simd_cast<int_v>(nPi2) & 3;

    const int_v sinSign = q << 30;
    const int_v cosSign = (q+1) << 30;

    const float_v sinS = sinSeries(x);
    const float_v cosS = cosSeries(x);

    const float_m mask = simd_cast<float_m>(q == 0 || q == 2);
    sinX = multiplySign( sinSign, iif(mask, sinS, cosS) );
    cosX = multiplySign( cosSign, iif(mask, cosS, sinS) );

    for(int i=0; i<4; i++) {
      if(fabs(sinX[i] - sin0[i]) > 1.e-7f * fabs(sin0[i]) || fabs(cosX[i] - cos0[i]) > 1.e-7f * fabs(cos0[i]) ) {
        std::cout << "x " << x << "   npi2 " << nPi2 << std::endl;
        std::cout << "sin " << sinX << "     " << sin0 << " " << sin(x) << std::endl;
        std::cout << "cos " << cosX << "     " << cos0 << " " << cos(x) << std::endl;
        std::cin.get();
      }
    }
#endif
}

static inline __attribute__((always_inline)) float_v ATan2( const float_v &y, const float_v &x )
{
    const float_v pi(3.1415926535897932f);
    const float_v zero(Vc::Zero);

    const float_m &xZero = (x == zero);
    const float_m &yZero = (y == zero);
    const float_m &xNeg  = (x < zero);
    const float_m &yNeg  = (y < zero);

    const float_v &absX = Vc::abs(x);
    const float_v &absY = Vc::abs(y);

    float_v a = absY / absX;
    const float_m &gt_tan_3pi_8 = (a > float_v(2.414213562373095f));
    const float_m &gt_tan_pi_8  = (a > float_v(0.4142135623730950f)) && (!gt_tan_3pi_8);
    float_v b(Vc::Zero);
    b(gt_tan_3pi_8) = pi/2.f;
    b(gt_tan_pi_8) = pi/4.f;
    a(gt_tan_3pi_8) =  (-1.f / a);
    a(gt_tan_pi_8) =  ((absY - absX) / (absY + absX)) ;
    const float_v &a2 = a * a;
    b += (((8.05374449538e-2f * a2
          - 1.38776856032E-1f) * a2
          + 1.99777106478E-1f) * a2
          - 3.33329491539E-1f) * a2 * a
          + a;
    b(xNeg ^ yNeg) = -b;
    b(xNeg && !yNeg) = (b+pi);
    b(xNeg &&  yNeg)  = (b-pi);
    b(xZero && yZero) = zero;
    b(xZero &&  yNeg) = (-pi/2.f);
    return b;
}

template<typename T>
struct ApplySin
{
    T operator()(const T& val) const
    {
        return std::sin(val);
    }
};

template<typename T>
struct ApplyCos
{
    T operator()(const T& val) const
    {
        return std::cos(val);
    }
};

int main()
{
    simd_float::value_type f1234[simd_float::SimdLen]{1.0f, 2.0f, 3.0f, 4.0f};

    simd_float sf1234;
    sf1234.load(f1234);
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print load of {1.0f, 2.0f, 3.0f, 4.0f} simd_float\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << sf1234 << '\n';
    Vc::float_v vf1234;
    vf1234.load(f1234);
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print load of {1.0f, 2.0f, 3.0f, 4.0f} Vc::float_v\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << vf1234 << '\n';

    // std::cout << std::string(20, '-') << '\n';
    // std::cout << "sin(1.0f), sin(2.0f), sin(3.0f), sin(4.0f):\n" ;
    // std::cout << std::string(20, '-') << '\n';
    // std::cout << std::sin(1.0f) << ", " ;
    // std::cout << std::sin(2.0f) << ", " ;
    // std::cout << std::sin(3.0f) << ", " ;
    // std::cout << std::sin(4.0f) << "\n" ;

    // std::cout << std::string(20, '-') << '\n';
    // std::cout << "apply sin to Print load_partial of {1.0f, 2.0f, 3.0f, 4.0f}\n" ;
    // std::cout << std::string(20, '-') << '\n';
    // std::cout << apply(simd_float{}.load_partial(5, f1234), ApplySin<float>{}) << "\n\n";

    // std::cout << std::string(20, '-') << '\n';
    // std::cout << "Sin to Print load_partial of {1.0f, 2.0f, 3.0f, 4.0f}\n" ;
    // std::cout << std::string(20, '-') << '\n';
    // std::cout << Sin(simd_float{}.load_partial(5, f1234)) << "\n\n";

    // std::cout << std::string(20, '-') << '\n';
    // std::cout << "SinVc to Print load of {1.0f, 2.0f, 3.0f, 4.0f}\n" ;
    // std::cout << std::string(20, '-') << '\n';
    // std::cout << SinVc(vf1234) << "\n\n";

    simd_float::value_type f5678[simd_float::SimdLen]{5.0f, 6.0f, 7.0f, 8.0f};

    simd_float sf5678, sfSin, sfCos;
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print load of {5.0f, 6.0f, 7.0f, 8.0f} simd_float\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << sf5678.load_partial(5, f5678) << '\n';

    float_v vf5678, vfSin, vfCos;
    vf5678.load(f5678) ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print load of {5.0f, 6.0f, 7.0f, 8.0f} Vc::float_v\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << vf5678 << '\n';

    std::cout << "\n\n";
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Expected scalar output of sin : ";
    std::cout << "sin(5.0f), sin(6.0f), sin(7.0f), sin(8.0f):\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << std::sin(5.0f) << ", " ;
    std::cout << std::sin(6.0f) << ", " ;
    std::cout << std::sin(7.0f) << ", " ;
    std::cout << std::sin(8.0f) << "\n" ;

    std::cout << std::string(20, '-') << '\n';
    std::cout << "Expected scalar output of cos : ";
    std::cout << "cos(5.0f), cos(6.0f), cos(7.0f), cos(8.0f):\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << std::cos(5.0f) << ", " ;
    std::cout << std::cos(6.0f) << ", " ;
    std::cout << std::cos(7.0f) << ", " ;
    std::cout << std::cos(8.0f) << "\n" ;

    std::cout << "\n\n";
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print sincos of {5.0f, 6.0f, 7.0f, 8.0f} Vc::float_v\n" ;
    std::cout << std::string(20, '-') << '\n';
    sincos(vf5678, vfSin, vfCos);
    std::cout << "Sin : " << vfSin << '\n';
    std::cout << "Cos : " << vfCos << '\n';

    std::cout << "\n\n";
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print sincos of {5.0f, 6.0f, 7.0f, 8.0f} simd_float\n" ;
    std::cout << std::string(20, '-') << '\n';
    sincos(sf5678, sfSin, sfCos);
    std::cout << "Sin : " << sfSin << '\n';
    std::cout << "Cos : " << sfCos << '\n';

    std::cout << "\n\n";
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print sincos of {5.0f, 6.0f, 7.0f, 8.0f} simd_float apply with functor\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Sin : " << KFP::SIMD::apply(sf5678, ApplySin<float>{}) << '\n';
    std::cout << "Cos : " << KFP::SIMD::apply(sf5678, ApplyCos<float>{}) << '\n';
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print sincos of {5.0f, 6.0f, 7.0f, 8.0f} simd_float apply with lambda\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Sin : " << KFP::SIMD::apply(sf5678, [](float x){return std::sin(x);}) << '\n';
    std::cout << "Cos : " << KFP::SIMD::apply(sf5678, [](float x){return std::cos(x);}) << '\n';

    std::cout << "\n\n";
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print atan2 of {1.0f, 2.0f, 3.0f, 4.0f} and {5.0f, 6.0f, 7.0f, 8.0f} Vc::float_v\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << "ATan2 : " << ATan2(vf1234, vf5678) << '\n';

    std::cout << "\n\n";
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print atan2 of {1.0f, 2.0f, 3.0f, 4.0f} and {5.0f, 6.0f, 7.0f, 8.0f} simd_float\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << "ATan2 : " << ATan2(sf1234, sf5678) << '\n';

    simd_float::value_type fp1234[simd_float::SimdLen]{0.1f, 0.2f, 0.3f, 0.4f};
    simd_float::value_type fp5678[simd_float::SimdLen]{0.5f, 0.6f, 0.7f, 0.8f};
    simd_float sfp1234{fp1234}, sfp5678{fp5678};
    float_v vfp1234{fp1234}, vfp5678{fp5678};
    std::cout << "\n\n";
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print asin of {0.1f, 0.2f, 0.3f, 0.4f} and {0.5f, 0.6f, 0.7f, 0.8f} simd_float apply with lambda\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Sin : " << KFP::SIMD::apply(sfp1234, [](float x){return std::asin(x);}) << '\n';
    std::cout << "Sin : " << KFP::SIMD::apply(sfp5678, [](float x){return std::asin(x);}) << '\n';
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print asin of {0.1f, 0.2f, 0.3f, 0.4f} and {0.5f, 0.6f, 0.7f, 0.8f} simd_float\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << ASin(sfp1234) << '\n';
    std::cout << ASin(sfp5678) << '\n';
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print asin of {0.1f, 0.2f, 0.3f, 0.4f} and {0.5f, 0.6f, 0.7f, 0.8f} Vc::float_v\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << Vc::asin(vfp1234) << '\n';
    std::cout << Vc::asin(vfp5678) << '\n';

    std::cout << "\n\n";
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print log of {1.0f, 2.0f, 3.0f, 4.0f} and {5.0f, 6.0f, 7.0f, 8.0f} simd_float\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << "First : " << Log(sf1234) << '\n';
    std::cout << "Second : " << Log(sf5678) << '\n';
    std::cout << std::string(20, '-') << '\n';
    std::cout << "Print log of {1.0f, 2.0f, 3.0f, 4.0f} and {5.0f, 6.0f, 7.0f, 8.0f} Vc::float_v\n" ;
    std::cout << std::string(20, '-') << '\n';
    std::cout << "First : " << Vc::log(vf1234) << '\n';
    std::cout << "Second : " << Vc::log(vf5678) << '\n';
}
