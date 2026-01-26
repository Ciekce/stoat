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

#include <array>

#include "../bitboard.h"
#include "../core.h"
#include "../util/multi_array.h"

#if ST_HAS_FAST_PEXT
    #include "sliders/bmi2.h"
#else
    #include "sliders/black_magic.h"
#endif

namespace stoat::attacks {
    namespace tables {
        consteval std::array<Bitboard, Squares::kCount> generateAttacks(auto func) {
            std::array<Bitboard, Squares::kCount> attacks{};

            for (i32 idx = 0; idx < Squares::kCount; ++idx) {
                func(attacks[idx], Square::fromRaw(idx));
                attacks[idx] &= Bitboards::kAll;
            }

            return attacks;
        }

        consteval std::array<std::array<Bitboard, Squares::kCount>, 2> generateSidedAttacks(auto func) {
            util::MultiArray<Bitboard, 2, Squares::kCount> attacks{};

            attacks[Colors::kBlack.idx()] =
                generateAttacks([&func](Bitboard& attacks, Square sq) { return func(Colors::kBlack, attacks, sq); });
            attacks[Colors::kWhite.idx()] =
                generateAttacks([&func](Bitboard& attacks, Square sq) { return func(Colors::kWhite, attacks, sq); });

            return attacks;
        }

        constexpr auto kPawnAttacks = generateSidedAttacks([](Color c, Bitboard& attacks, Square sq) {
            const auto bit = Bitboard::fromSquare(sq);
            attacks |= bit.shiftNorthRelative(c);
        });

        constexpr auto kKnightAttacks = generateSidedAttacks([](Color c, Bitboard& attacks, Square sq) {
            const auto bit = Bitboard::fromSquare(sq);

            attacks |= bit.shiftNorthRelative(c).shiftNorthWestRelative(c);
            attacks |= bit.shiftNorthRelative(c).shiftNorthEastRelative(c);
        });

        constexpr auto kSilverAttacks = generateSidedAttacks([](Color c, Bitboard& attacks, Square sq) {
            const auto bit = Bitboard::fromSquare(sq);

            attacks |= bit.shiftNorthWest();
            attacks |= bit.shiftNorthEast();
            attacks |= bit.shiftSouthWest();
            attacks |= bit.shiftSouthEast();

            attacks |= bit.shiftNorthRelative(c);
        });

        constexpr auto kGoldAttacks = generateSidedAttacks([](Color c, Bitboard& attacks, Square sq) {
            const auto bit = Bitboard::fromSquare(sq);

            attacks |= bit.shiftNorth();
            attacks |= bit.shiftSouth();
            attacks |= bit.shiftWest();
            attacks |= bit.shiftEast();

            attacks |= bit.shiftNorthWestRelative(c);
            attacks |= bit.shiftNorthEastRelative(c);
        });

        constexpr auto kKingAttacks = generateAttacks([](Bitboard& attacks, Square sq) {
            const auto bit = Bitboard::fromSquare(sq);

            attacks |= bit.shiftNorth();
            attacks |= bit.shiftSouth();
            attacks |= bit.shiftWest();
            attacks |= bit.shiftEast();

            attacks |= bit.shiftNorthWest();
            attacks |= bit.shiftNorthEast();
            attacks |= bit.shiftSouthWest();
            attacks |= bit.shiftSouthEast();
        });
    } // namespace tables

    [[nodiscard]] constexpr Bitboard pawnAttacks(Square sq, Color c) {
        assert(sq);
        assert(c);
        return tables::kPawnAttacks[c.idx()][sq.idx()];
    }

    [[nodiscard]] constexpr Bitboard lanceAttacks(Square sq, Color c, Bitboard occ) {
        assert(sq);
        assert(c);

        if (std::is_constant_evaluated()) {
            if (c == Colors::kBlack) {
                return sliders::generateMultiSlidingAttacks<offsets::kNorth>(sq, occ);
            } else {
                return sliders::generateMultiSlidingAttacks<offsets::kSouth>(sq, occ);
            }
        }

        return sliders::lanceAttacks(sq, c, occ);
    }

    [[nodiscard]] constexpr Bitboard knightAttacks(Square sq, Color c) {
        assert(sq);
        assert(c);
        return tables::kKnightAttacks[c.idx()][sq.idx()];
    }

    [[nodiscard]] constexpr Bitboard silverAttacks(Square sq, Color c) {
        assert(sq);
        assert(c);
        return tables::kSilverAttacks[c.idx()][sq.idx()];
    }

    [[nodiscard]] constexpr Bitboard goldAttacks(Square sq, Color c) {
        assert(sq);
        assert(c);
        return tables::kGoldAttacks[c.idx()][sq.idx()];
    }

    [[nodiscard]] constexpr Bitboard bishopAttacks(Square sq, Bitboard occ) {
        assert(sq);

        if (std::is_constant_evaluated()) {
            return sliders::generateMultiSlidingAttacks<
                offsets::kNorthWest,
                offsets::kNorthEast,
                offsets::kSouthWest,
                offsets::kSouthEast>(sq, occ);
        }

        return sliders::bishopAttacks(sq, occ);
    }

    [[nodiscard]] constexpr Bitboard rookAttacks(Square sq, Bitboard occ) {
        assert(sq);

        if (std::is_constant_evaluated()) {
            return sliders::
                generateMultiSlidingAttacks<offsets::kNorth, offsets::kSouth, offsets::kWest, offsets::kEast>(sq, occ);
        }

        return sliders::rookAttacks(sq, occ);
    }

    [[nodiscard]] constexpr Bitboard kingAttacks(Square sq) {
        assert(sq);
        return tables::kKingAttacks[sq.idx()];
    }

    [[nodiscard]] constexpr Bitboard promotedBishopAttacks(Square sq, Bitboard occ) {
        assert(sq);
        return bishopAttacks(sq, occ) | kingAttacks(sq);
    }

    [[nodiscard]] constexpr Bitboard promotedRookAttacks(Square sq, Bitboard occ) {
        assert(sq);
        return rookAttacks(sq, occ) | kingAttacks(sq);
    }

    [[nodiscard]] constexpr Bitboard adjacentAttacks(PieceType pt, Square sq, Color c) {
        assert(pt);
        assert(sq);
        assert(c);

        switch (pt.raw()) {
            case PieceTypes::kPawn.raw():
            case PieceTypes::kLance.raw():
                return pawnAttacks(sq, c);
            case PieceTypes::kPromotedPawn.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kKnight.raw():
                return knightAttacks(sq, c);
            case PieceTypes::kPromotedLance.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kPromotedKnight.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kSilver.raw():
                return silverAttacks(sq, c);
            case PieceTypes::kPromotedSilver.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kGold.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kBishop.raw():
            case PieceTypes::kRook.raw():
                return Bitboards::kEmpty;
            case PieceTypes::kPromotedBishop.raw():
            case PieceTypes::kPromotedRook.raw():
            case PieceTypes::kKing.raw():
                return kingAttacks(sq);
        }

        __builtin_unreachable();
    }

    [[nodiscard]] constexpr Bitboard pieceAttacks(PieceType pt, Square sq, Color c, Bitboard occ) {
        assert(pt);
        assert(sq);
        assert(c);

        switch (pt.raw()) {
            case PieceTypes::kPawn.raw():
                return pawnAttacks(sq, c);
            case PieceTypes::kLance.raw():
                return lanceAttacks(sq, c, occ);
            case PieceTypes::kKnight.raw():
                return knightAttacks(sq, c);
            case PieceTypes::kSilver.raw():
                return silverAttacks(sq, c);
            case PieceTypes::kBishop.raw():
                return bishopAttacks(sq, occ);
            case PieceTypes::kRook.raw():
                return rookAttacks(sq, occ);
            case PieceTypes::kGold.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kKing.raw():
                return kingAttacks(sq);
            case PieceTypes::kPromotedPawn.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kPromotedLance.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kPromotedKnight.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kPromotedSilver.raw():
                return goldAttacks(sq, c);
            case PieceTypes::kPromotedBishop.raw():
                return promotedBishopAttacks(sq, occ);
            case PieceTypes::kPromotedRook.raw():
                return promotedRookAttacks(sq, occ);
        }

        __builtin_unreachable();
    }
} // namespace stoat::attacks
