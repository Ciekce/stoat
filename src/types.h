/*
 * Stoat, a USI shogi engine
 * Copyright (C) 2025 Ciekce
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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>

#include <fmt/format.h>

namespace stoat {
    using u8 = std::uint8_t;
    using u16 = std::uint16_t;
    using u32 = std::uint32_t;
    using u64 = std::uint64_t;
    using u128 = unsigned __int128;

    using i8 = std::int8_t;
    using i16 = std::int16_t;
    using i32 = std::int32_t;
    using i64 = std::int64_t;
    using i128 = __int128;

    using f32 = float;
    using f64 = double;

    using usize = std::size_t;

    [[nodiscard]] constexpr u128 toU128(u64 high, u64 low) {
        return (static_cast<u128>(high) << 64) | static_cast<u128>(low);
    }

    [[nodiscard]] constexpr std::pair<u64, u64> fromU128(u128 v) {
        return {static_cast<u64>(v >> 64), static_cast<u64>(v)};
    }
} // namespace stoat

#define U128(High, Low) (stoat::toU128(UINT64_C(High), UINT64_C(Low)))

#define ST_STRINGIFY_(S) #S
#define ST_STRINGIFY(S) ST_STRINGIFY_(S)

#define ST_UNUSED(...) (void)sizeof(__VA_ARGS__)
