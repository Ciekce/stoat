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

#include "../../arch.h"

#if !ST_HAS_FAST_PEXT
    #include "black_magic.h"

namespace stoat::attacks::sliders::black_magic {
    namespace {
        template <usize kTableSize, i32... kDirs>
        std::array<Bitboard, kTableSize> generateAttacks(
            const internal::PieceData& data,
            std::span<const u128, Squares::kCount> magics,
            std::span<const i32, Squares::kCount> shifts
        ) {
            std::array<Bitboard, kTableSize> dst{};

            for (i32 sqIdx = 0; sqIdx < Squares::kCount; ++sqIdx) {
                const auto sq = Square::fromRaw(sqIdx);
                const auto& sqData = data.squares[sq.idx()];

                const auto magic = magics[sq.idx()];
                const auto shift = shifts[sq.idx()];

                const auto maxEntries = sqData.mask == 0 ? 1 : 1 << util::popcount(~sqData.mask);

                for (i32 i = 0; i < maxEntries; ++i) {
                    const auto occ = Bitboard{util::pdep(i, ~sqData.mask)};
                    const auto idx = calcIdx(occ, sqData.mask, magic, shift);

                    if (!dst[sqData.offset + idx].empty()) {
                        continue;
                    }

                    auto& attacks = dst[sqData.offset + idx];

                    for (const auto dir : {kDirs...}) {
                        attacks |= internal::generateSlidingAttacks(sq, dir, occ);
                    }
                }
            }

            return dst;
        }
    } // namespace

    const util::MultiArray<Bitboard, Colors::kCount, kLanceDataTableSize> g_lanceAttacks = {
        generateAttacks<kLanceDataTableSize, offsets::kNorth>(
            lanceData(Colors::kBlack),
            lanceMagics(Colors::kBlack),
            lanceShifts(Colors::kBlack)
        ),
        generateAttacks<kLanceDataTableSize, offsets::kSouth>(
            lanceData(Colors::kWhite),
            lanceMagics(Colors::kWhite),
            lanceShifts(Colors::kWhite)
        ),
    };

    const std::array<Bitboard, kBishopData.tableSize> g_bishopAttacks = generateAttacks<
        kBishopData.tableSize,
        offsets::kNorthWest,
        offsets::kNorthEast,
        offsets::kSouthWest,
        offsets::kSouthEast>(kBishopData, kBishopMagics, kBishopShifts);

    const std::array<Bitboard, kRookData.tableSize> g_rookAttacks =
        generateAttacks<kRookData.tableSize, offsets::kNorth, offsets::kSouth, offsets::kWest, offsets::kEast>(
            kRookData,
            kRookMagics,
            kRookShifts
        );
} // namespace stoat::attacks::sliders::black_magic
#endif
