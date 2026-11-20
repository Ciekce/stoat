/*
 * Stoat, a USI shogi engine
 * Copyright (C) 2026 Ciekce
 *
 * Stoat is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Stoat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Stoat. If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "../../types.h"

#include <immintrin.h>

namespace stoat::eval::nnue::simd {
    using VecU8 = __m256i;
    using VecI8 = __m256i;
    using VecI16 = __m256i;
    using VecI32 = __m256i;

    constexpr usize kChunkSize8 = sizeof(VecI8) / sizeof(i8);
    constexpr usize kChunkSize16 = sizeof(VecI16) / sizeof(i16);
    constexpr usize kChunkSize32 = sizeof(VecI32) / sizeof(i32);

    // ================================ u8 ================================

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecU8 loadU8(const u8* ptr) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    ST_ALWAYS_INLINE_NDEBUG void storeU8(u8* ptr, VecU8 v) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecU8 zeroU8() {
        return _mm256_setzero_si256();
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecU8 set1U8(u8 v) {
        return _mm256_set1_epi8(static_cast<i8>(v));
    }

    // ================================ i8 ================================

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI8 loadI8(const i8* ptr) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    ST_ALWAYS_INLINE_NDEBUG void storeI8(i8* ptr, VecI8 v) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI8 zeroI8() {
        return _mm256_setzero_si256();
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI8 set1I8(i8 v) {
        return _mm256_set1_epi8(v);
    }

    // ================================ i16 ================================

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 loadI16(const i16* ptr) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    ST_ALWAYS_INLINE_NDEBUG void storeI16(i16* ptr, VecI16 v) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 zeroI16() {
        return _mm256_setzero_si256();
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 set1I16(i16 v) {
        return _mm256_set1_epi16(v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 minI16(VecI16 a, VecI16 b) {
        return _mm256_min_epi16(a, b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 maxI16(VecI16 a, VecI16 b) {
        return _mm256_max_epi16(a, b);
    }

    template <u32 kShift>
    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 mulhiShiftI16(VecI16 a, VecI16 b) {
        return _mm256_mulhi_epi16(_mm256_slli_epi16(a, kShift), b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecU8 packusI16(VecI16 a, VecI16 b) {
        const auto packed = _mm256_packus_epi16(a, b);
        return _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
    }

    // ================================ i32 ================================

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 loadI32(const i32* ptr) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    ST_ALWAYS_INLINE_NDEBUG void storeI32(i32* ptr, VecI32 v) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 zeroI32() {
        return _mm256_setzero_si256();
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 set1I32(i32 v) {
        return _mm256_set1_epi32(v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 addI32(VecI32 a, VecI32 b) {
        return _mm256_add_epi32(a, b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 mulI32(VecI32 a, VecI32 b) {
        return _mm256_mullo_epi32(a, b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 minI32(VecI32 a, VecI32 b) {
        return _mm256_min_epi32(a, b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 maxI32(VecI32 a, VecI32 b) {
        return _mm256_max_epi32(a, b);
    }

    template <u32 kShift>
    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 shiftLeftI32(VecI32 v) {
        return _mm256_slli_epi32(v, kShift);
    }

    template <u32 kShift>
    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 shiftRightI32(VecI32 v) {
        return _mm256_srai_epi32(v, kShift);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 dotU8I8I32(VecI32 acc, VecU8 u, VecI8 i) {
        const auto p = _mm256_maddubs_epi16(u, i);
        const auto w = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
        return _mm256_add_epi32(acc, w);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG i32 hsumI32(VecI32 v) {
        const auto hi128 = _mm256_extracti128_si256(v, 1);
        const auto lo128 = _mm256_castsi256_si128(v);

        const auto sum128 = _mm_add_epi32(hi128, lo128);

        const auto hi64 = _mm_unpackhi_epi64(sum128, sum128);
        const auto sum64 = _mm_add_epi32(sum128, hi64);

        const auto hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        const auto sum32 = _mm_add_epi32(sum64, hi32);

        return _mm_cvtsi128_si32(sum32);
    }
}
