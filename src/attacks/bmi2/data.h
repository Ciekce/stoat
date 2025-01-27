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

#include "../../bitboard.h"
#include "../../core.h"
#include "../util.h"

namespace stoat::attacks::sliders::bmi2 {
    namespace internal {
        struct SquareData {
            Bitboard blockerMask;
            Bitboard attackMask;

            u32 offset;

            i32 blockerShift;
            i32 attackShift;
        };

        struct PieceData {
            std::array<SquareData, Squares::kCount> squares;
            u32 tableSize;
        };

        template <i32... Dirs>
        consteval PieceData generatePieceData() {
            PieceData dst{};

            for (i32 sqIdx = 0; sqIdx < Squares::kCount; ++sqIdx) {
                const auto sq = Square::fromRaw(sqIdx);
                auto& sqData = dst.squares[sq.idx()];

                for (const auto dir : {Dirs...}) {
                    const auto attacks = attacks::internal::generateSlidingAttacks(sq, dir, Bitboards::kEmpty);

                    sqData.blockerMask |= attacks & ~attacks::internal::edges(dir);
                    sqData.attackMask |= attacks;
                }

                sqData.offset = dst.tableSize;

                sqData.blockerShift = std::popcount(static_cast<u64>(sqData.blockerMask.raw()));
                sqData.attackShift = std::popcount(static_cast<u64>(sqData.attackMask.raw()));

                dst.tableSize += 1 << sqData.blockerMask.popcount();
            }

            return dst;
        }
    } // namespace internal

    constexpr std::array<internal::PieceData, Colors::kCount> kLanceData = {
        internal::generatePieceData<offsets::kNorth>(), // black
        internal::generatePieceData<offsets::kSouth>(), // white
    };

    static_assert(kLanceData[0].tableSize == kLanceData[1].tableSize);

    [[nodiscard]] constexpr const internal::PieceData& lanceData(Color c) {
        assert(c);
        return kLanceData[c.idx()];
    }

    constexpr usize kLanceDataTableSize = kLanceData[0].tableSize;

    constexpr auto kBishopData =
        internal::generatePieceData<offsets::kNorthWest, offsets::kNorthEast, offsets::kSouthWest, offsets::kSouthEast>(
        );

    constexpr auto kRookData =
        internal::generatePieceData<offsets::kNorth, offsets::kSouth, offsets::kWest, offsets::kEast>();
} // namespace stoat::attacks::sliders::bmi2
