#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif

namespace throne_simd {

static inline void copy_i16(int16_t* dst, const int16_t* src, std::size_t n) {
#if defined(__AVX2__) || defined(_M_AVX2)
    std::size_t i = 0;
    const std::size_t n16 = n & ~std::size_t(15);
    for (; i < n16; i += 16) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }

    for (; i < n; ++i)
        dst[i] = src[i];
#else
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = src[i];
#endif
}

static inline void add_i16(int16_t* dst, const int16_t* src, std::size_t n) {
#if defined(__AVX2__) || defined(_M_AVX2)
    std::size_t i = 0;
    const std::size_t n16 = n & ~std::size_t(15);
    for (; i < n16; i += 16) {
        const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi16(d, s));
    }

    for (; i < n; ++i)
        dst[i] = static_cast<int16_t>(dst[i] + src[i]);
#else
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = static_cast<int16_t>(dst[i] + src[i]);
#endif
}

static inline void sub_i16(int16_t* dst, const int16_t* src, std::size_t n) {
#if defined(__AVX2__) || defined(_M_AVX2)
    std::size_t i = 0;
    const std::size_t n16 = n & ~std::size_t(15);
    for (; i < n16; i += 16) {
        const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_sub_epi16(d, s));
    }

    for (; i < n; ++i)
        dst[i] = static_cast<int16_t>(dst[i] - src[i]);
#else
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = static_cast<int16_t>(dst[i] - src[i]);
#endif
}

static inline void add_sub_i16(int16_t* dst, const int16_t* add, const int16_t* sub, std::size_t n) {
#if defined(__AVX2__) || defined(_M_AVX2)
    std::size_t i = 0;
    const std::size_t n16 = n & ~std::size_t(15);
    for (; i < n16; i += 16) {
        __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(add + i));
        const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sub + i));
        d = _mm256_add_epi16(d, a);
        d = _mm256_sub_epi16(d, s);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), d);
    }

    for (; i < n; ++i)
        dst[i] = static_cast<int16_t>(dst[i] + add[i] - sub[i]);
#else
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = static_cast<int16_t>(dst[i] + add[i] - sub[i]);
#endif
}

static inline int32_t dot_u8_i8(const uint8_t* a, const int8_t* b, std::size_t n) {
#if defined(__AVX2__) || defined(_M_AVX2)
    std::size_t i = 0;
    __m256i acc32 = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);
    const std::size_t n32 = n & ~std::size_t(31);

    for (; i < n32; i += 32) {
        const __m256i va8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

        const __m128i va_lo8 = _mm256_castsi256_si128(va8);
        const __m128i va_hi8 = _mm256_extracti128_si256(va8, 1);
        const __m128i vb_lo8 = _mm256_castsi256_si128(vb8);
        const __m128i vb_hi8 = _mm256_extracti128_si256(vb8, 1);

        const __m256i va_lo16 = _mm256_cvtepu8_epi16(va_lo8);
        const __m256i va_hi16 = _mm256_cvtepu8_epi16(va_hi8);
        const __m256i vb_lo16 = _mm256_cvtepi8_epi16(vb_lo8);
        const __m256i vb_hi16 = _mm256_cvtepi8_epi16(vb_hi8);

        const __m256i prod_lo16 = _mm256_mullo_epi16(va_lo16, vb_lo16);
        const __m256i prod_hi16 = _mm256_mullo_epi16(va_hi16, vb_hi16);

        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(prod_lo16, ones16));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(prod_hi16, ones16));
    }

    __m128i lo = _mm256_castsi256_si128(acc32);
    __m128i hi = _mm256_extracti128_si256(acc32, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    int32_t out = _mm_cvtsi128_si32(sum);

    for (; i < n; ++i)
        out += int32_t(a[i]) * int32_t(b[i]);

    return out;
#else
    int32_t out = 0;
    for (std::size_t i = 0; i < n; ++i)
        out += int32_t(a[i]) * int32_t(b[i]);
    return out;
#endif
}

} // namespace throne_simd
