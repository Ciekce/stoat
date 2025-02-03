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

#include "types.h"

#include "core.h"
#include "position.h"

namespace stoat::see {
    namespace values {
        constexpr i32 kPawn = 100;
        constexpr i32 kPromotedPawn = 1000;
        constexpr i32 kLance = 400;
        constexpr i32 kKnight = 500;
        constexpr i32 kPromotedLance = 900;
        constexpr i32 kPromotedKnight = 900;
        constexpr i32 kSilver = 600;
        constexpr i32 kPromotedSilver = 800;
        constexpr i32 kGold = 800;
        constexpr i32 kBishop = 1100;
        constexpr i32 kRook = 1300;
        constexpr i32 kPromotedBishop = 1500;
        constexpr i32 kPromotedRook = 1700;
        constexpr i32 kKing = 0;
    } // namespace values

    [[nodiscard]] constexpr i32 pieceValue(PieceType pt) {
        constexpr std::array kValues = {
            values::kPawn,
            values::kPromotedPawn,
            values::kLance,
            values::kKnight,
            values::kPromotedLance,
            values::kPromotedKnight,
            values::kSilver,
            values::kPromotedSilver,
            values::kGold,
            values::kBishop,
            values::kRook,
            values::kPromotedBishop,
            values::kPromotedRook,
            values::kKing,
            0, // none
        };

        assert(pt);
        return kValues[pt.idx()];
    }

    [[nodiscard]] bool see(const Position& pos, Move move, i32 threshold);
} // namespace stoat::see
