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

#include "../types.h"

#include "../core.h"

namespace stoat::eval {
    namespace values {
        constexpr Score kPawn = 100;
        constexpr Score kPromotedPawn = 1000;
        constexpr Score kLance = 400;
        constexpr Score kKnight = 500;
        constexpr Score kPromotedLance = 900;
        constexpr Score kPromotedKnight = 900;
        constexpr Score kSilver = 600;
        constexpr Score kPromotedSilver = 800;
        constexpr Score kGold = 800;
        constexpr Score kBishop = 1100;
        constexpr Score kRook = 1300;
        constexpr Score kPromotedBishop = 1500;
        constexpr Score kPromotedRook = 1700;
    } // namespace values

    [[nodiscard]] constexpr Score pieceValue(PieceType pt) {
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
        };

        assert(pt);
        return kValues[pt.idx()];
    }
} // namespace stoat::eval
