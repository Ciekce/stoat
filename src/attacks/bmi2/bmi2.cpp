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

#if ST_HAS_FAST_PEXT
    #include "bmi2.h"

namespace stoat::attacks::sliders::bmi2 {
    namespace {
        template <typename CompressedType, const internal::PieceData& kData, i32... Dirs>
        std::array<CompressedType, kData.tableSize> generateAttacks() {
            std::array<CompressedType, kData.tableSize> dst{};

            for (i32 sqIdx = 0; sqIdx < Squares::kCount; ++sqIdx) {
                const auto sq = Square::fromRaw(sqIdx);
                const auto& sqData = kData.squares[sq.idx()];

                const auto entries = 1 << sqData.blockerMask.popcount();

                for (i32 i = 0; i < entries; ++i) {
                    const auto occ = Bitboard{util::pdep(i, sqData.blockerMask.raw())};

                    Bitboard attacks{};

                    //auto& attacks = ;

                    for (const auto dir : {Dirs...}) {
                        attacks |= attacks::internal::generateSlidingAttacks(sq, dir, occ);
                    }

                    const auto compressed = util::pext(attacks.raw(), sqData.attackMask.raw(), sqData.attackShift);
                    dst[sqData.offset + i] = static_cast<CompressedType>(compressed);

                    assert(static_cast<u128>(dst[sqData.offset + i]) == compressed);
                }
            }

            return dst;
        }
    } // namespace

    const util::MultiArray<u8, Colors::kCount, kLanceDataTableSize> g_lanceAttacks = {
        generateAttacks<u8, lanceData(Colors::kBlack), offsets::kNorth>(),
        generateAttacks<u8, lanceData(Colors::kWhite), offsets::kSouth>(),
    };

    const std::array<u16, kBishopData.tableSize> g_bishopAttacks = generateAttacks<
        u16,
        kBishopData,
        offsets::kNorthWest,
        offsets::kNorthEast,
        offsets::kSouthWest,
        offsets::kSouthEast>();

    const std::array<u16, kRookData.tableSize> g_rookAttacks =
        generateAttacks<u16, kRookData, offsets::kNorth, offsets::kSouth, offsets::kWest, offsets::kEast>();
} // namespace stoat::attacks::sliders::bmi2

#endif
