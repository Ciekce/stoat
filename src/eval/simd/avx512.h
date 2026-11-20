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
    using VecU8 = __m512i;
    using VecI8 = __m512i;
    using VecI16 = __m512i;
    using VecI32 = __m512i;

    constexpr usize kChunkSize8 = sizeof(VecI8) / sizeof(i8);
    constexpr usize kChunkSize16 = sizeof(VecI16) / sizeof(i16);
    constexpr usize kChunkSize32 = sizeof(VecI32) / sizeof(i32);

    // ================================ u8 ================================

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecU8 loadU8(const u8* ptr) {
        return _mm512_load_si512(ptr);
    }

    ST_ALWAYS_INLINE_NDEBUG void storeU8(u8* ptr, VecU8 v) {
        _mm512_store_si512(ptr, v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecU8 zeroU8() {
        return _mm512_setzero_si512();
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecU8 set1U8(u8 v) {
        return _mm512_set1_epi8(static_cast<i8>(v));
    }

    // ================================ i8 ================================

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI8 loadI8(const i8* ptr) {
        return _mm512_load_si512(ptr);
    }

    ST_ALWAYS_INLINE_NDEBUG void storeI8(i8* ptr, VecI8 v) {
        _mm512_store_si512(ptr, v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI8 zeroI8() {
        return _mm512_setzero_si512();
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI8 set1I8(i8 v) {
        return _mm512_set1_epi8(v);
    }

    // ================================ i16 ================================

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 loadI16(const i16* ptr) {
        return _mm512_load_si512(ptr);
    }

    ST_ALWAYS_INLINE_NDEBUG void storeI16(i16* ptr, VecI16 v) {
        _mm512_store_si512(ptr, v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 zeroI16() {
        return _mm512_setzero_si512();
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 set1I16(i16 v) {
        return _mm512_set1_epi16(v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 minI16(VecI16 a, VecI16 b) {
        return _mm512_min_epi16(a, b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 maxI16(VecI16 a, VecI16 b) {
        return _mm512_max_epi16(a, b);
    }

    template <u32 kShift>
    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI16 mulhiShiftI16(VecI16 a, VecI16 b) {
        return _mm512_mulhi_epi16(_mm512_slli_epi16(a, kShift), b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecU8 packusI16(VecI16 a, VecI16 b) {
        const auto packed = _mm512_packus_epi16(a, b);
        return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), packed);
    }

    // ================================ i32 ================================

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 loadI32(const i32* ptr) {
        return _mm512_load_si512(ptr);
    }

    ST_ALWAYS_INLINE_NDEBUG void storeI32(i32* ptr, VecI32 v) {
        _mm512_store_si512(ptr, v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 zeroI32() {
        return _mm512_setzero_si512();
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 set1I32(i32 v) {
        return _mm512_set1_epi32(v);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 addI32(VecI32 a, VecI32 b) {
        return _mm512_add_epi32(a, b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 mulI32(VecI32 a, VecI32 b) {
        return _mm512_mullo_epi32(a, b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 minI32(VecI32 a, VecI32 b) {
        return _mm512_min_epi32(a, b);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 maxI32(VecI32 a, VecI32 b) {
        return _mm512_max_epi32(a, b);
    }

    template <u32 kShift>
    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 shiftLeftI32(VecI32 v) {
        return _mm512_slli_epi32(v, kShift);
    }

    template <u32 kShift>
    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 shiftRightI32(VecI32 v) {
        return _mm512_srai_epi32(v, kShift);
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG VecI32 dotU8I8I32(VecI32 acc, VecU8 u, VecI8 i) {
#if __AVX512VNNI__
        return _mm512_dpbusd_epi32(acc, u, i);
#else
        const auto p = _mm512_maddubs_epi16(u, i);
        const auto w = _mm512_madd_epi16(p, _mm512_set1_epi16(1));
        return _mm512_add_epi32(acc, w);
#endif
    }

    [[nodiscard]] ST_ALWAYS_INLINE_NDEBUG i32 hsumI32(VecI32 v) {
        return _mm512_reduce_add_epi32(v);
    }
} // namespace stoat::eval::nnue::simd
