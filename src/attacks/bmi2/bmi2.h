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

#include "../../types.h"

#include <array>
#include <span>

#include "../../core.h"
#include "../../util/bits.h"
#include "../../util/multi_array.h"
#include "data.h"

namespace stoat::attacks::sliders {
    namespace bmi2 {
        extern const util::MultiArray<u8, Colors::kCount, kLanceDataTableSize> g_lanceAttacks;

        [[nodiscard]] inline std::span<const u8, kLanceDataTableSize> lanceAttacks(Color c) {
            assert(c);
            return g_lanceAttacks[c.idx()];
        }

        extern const std::array<u16, kBishopData.tableSize> g_bishopAttacks;
        extern const std::array<u16, kRookData.tableSize> g_rookAttacks;
    } // namespace bmi2

    [[nodiscard]] inline Bitboard lanceAttacks(Square sq, Color c, Bitboard occ) {
        const auto& sqData = bmi2::lanceData(c).squares[sq.idx()];

        const usize idx = util::pext(occ.raw(), sqData.blockerMask.raw(), sqData.blockerShift);
        const auto compressed = bmi2::lanceAttacks(c)[sqData.offset + idx];

        return Bitboard{util::pdep(compressed, sqData.attackMask.raw(), sqData.attackShift)};
    }

    [[nodiscard]] inline Bitboard bishopAttacks(Square sq, Bitboard occ) {
        const auto& sqData = bmi2::kBishopData.squares[sq.idx()];

        const usize idx = util::pext(occ.raw(), sqData.blockerMask.raw(), sqData.blockerShift);
        const auto compressed = bmi2::g_bishopAttacks[sqData.offset + idx];

        return Bitboard{util::pdep(compressed, sqData.attackMask.raw(), sqData.attackShift)};
    }

    [[nodiscard]] inline Bitboard rookAttacks(Square sq, Bitboard occ) {
        const auto& sqData = bmi2::kRookData.squares[sq.idx()];

        const usize idx = util::pext(occ.raw(), sqData.blockerMask.raw(), sqData.blockerShift);
        const auto compressed = bmi2::g_rookAttacks[sqData.offset + idx];

        return Bitboard{util::pdep(compressed, sqData.attackMask.raw(), sqData.attackShift)};
    }
} // namespace stoat::attacks::sliders
