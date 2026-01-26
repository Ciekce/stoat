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

#include <array>
#include <cassert>

#include "core.h"
#include "util/result.h"

namespace stoat {
    struct InvalidMoveError {};

    class Move {
    public:
        constexpr Move() = default;

        constexpr Move(const Move&) = default;
        constexpr Move(Move&&) = default;

        [[nodiscard]] constexpr u16 raw() const {
            return m_move;
        }

        [[nodiscard]] constexpr bool isDrop() const {
            return get(kDropFlagShift, kFlagMask) != 0;
        }

        [[nodiscard]] constexpr bool isPromo() const {
            assert(!isDrop());
            return get(kNormalPromoFlagShift, kFlagMask) != 0;
        }

        [[nodiscard]] constexpr Square from() const {
            assert(!isDrop());
            return Square::fromRaw(get(kNormalFromShift, kSquareMask));
        }

        [[nodiscard]] constexpr Square to() const {
            return Square::fromRaw(get(kToShift, kSquareMask));
        }

        [[nodiscard]] constexpr PieceType dropPiece() const {
            constexpr std::array kDropPieces = {
                PieceTypes::kPawn,
                PieceTypes::kLance,
                PieceTypes::kKnight,
                PieceTypes::kSilver,
                PieceTypes::kGold,
                PieceTypes::kBishop,
                PieceTypes::kRook,
            };

            assert(isDrop());

            return kDropPieces[get(kDropPieceShift, kPieceMask)];
        }

        [[nodiscard]] constexpr bool isNull() const {
            return m_move == 0;
        }

        [[nodiscard]] constexpr explicit operator bool() const {
            return !isNull();
        }

        [[nodiscard]] constexpr bool operator==(const Move&) const = default;

        constexpr Move& operator=(const Move&) = default;
        constexpr Move& operator=(Move&&) = default;

        [[nodiscard]] static constexpr Move fromRaw(u16 raw) {
            return Move{raw};
        }

        [[nodiscard]] static constexpr Move makeNormal(Square from, Square to) {
            assert(from);
            assert(to);

            return Move{static_cast<u16>((to.raw() << kToShift) | (from.raw() << kNormalFromShift))};
        }

        [[nodiscard]] static constexpr Move makePromotion(Square from, Square to) {
            assert(from);
            assert(to);

            return Move{static_cast<u16>(
                (to.raw() << kToShift) | (from.raw() << kNormalFromShift) | (1 << kNormalPromoFlagShift)
            )};
        }

        [[nodiscard]] static constexpr Move makeDrop(PieceType piece, Square to) {
            constexpr std::array kDropPieceIndices = {
                0, // pawn
                -1,
                1, // lance
                2, // knight
                -1,
                -1,
                3, // silver
                -1,
                4, // gold
                5, // bishop
                6, // rook
            };

            assert(piece);
            assert(!piece.isPromoted());
            assert(piece != PieceTypes::kKing);
            assert(to);

            const auto pieceIdx = kDropPieceIndices[piece.idx()];
            assert(pieceIdx != -1);

            return Move{
                static_cast<u16>((to.raw() << kToShift) | (pieceIdx << kDropPieceShift) | (1 << kDropFlagShift))
            };
        }

        [[nodiscard]] static constexpr util::Result<Move, InvalidMoveError> fromStr(std::string_view str) {
            if (str.size() < 4 || str.size() > 5) {
                return util::err<InvalidMoveError>();
            }

            if (str[1] == '*') {
                if (str.size() != 4) {
                    return util::err<InvalidMoveError>();
                }

                const auto piece = PieceType::unpromotedFromChar(str[0]);
                const auto square = Square::fromStr(str.substr(2, 2));

                if (!piece || !square || piece == PieceTypes::kKing) {
                    return util::err<InvalidMoveError>();
                }

                return util::ok(makeDrop(piece, square));
            }

            if (str.size() == 5 && str[4] != '+') {
                return util::err<InvalidMoveError>();
            }

            const bool promo = str.size() == 5;

            const auto from = Square::fromStr(str.substr(0, 2));
            const auto to = Square::fromStr(str.substr(2, 2));

            if (!from || !to) {
                return util::err<InvalidMoveError>();
            }

            return util::ok(promo ? makePromotion(from, to) : makeNormal(from, to));
        }

    private:
        static constexpr i32 kToShift = 0;

        static constexpr i32 kNormalFromShift = 7;
        static constexpr i32 kNormalPromoFlagShift = 14;

        static constexpr i32 kDropPieceShift = 7;

        static constexpr i32 kDropFlagShift = 15;

        static constexpr u16 kSquareMask = 0b1111111;
        static constexpr u16 kPieceMask = 0b111;
        static constexpr u16 kFlagMask = 0b1;

        explicit constexpr Move(u16 move) :
                m_move{move} {}

        u16 m_move;

        [[nodiscard]] constexpr u16 get(i32 shift, u16 mask) const {
            return (m_move >> shift) & mask;
        }
    };

    constexpr Move kNullMove{};
} // namespace stoat

template <>
struct fmt::formatter<stoat::Move> : fmt::formatter<std::string_view> {
    constexpr format_context::iterator format(stoat::Move value, format_context& ctx) const {
        if (value.isNull()) {
            return format_to(ctx.out(), "????");
        }

        if (value.isDrop()) {
            const auto piece = value.dropPiece();
            const auto square = value.to();

            return format_to(ctx.out(), "{}*{}", piece, square);
        }

        const auto from = value.from();
        const auto to = value.to();

        if (value.isPromo()) {
            return format_to(ctx.out(), "{}{}+", from, to);
        } else {
            return format_to(ctx.out(), "{}{}", from, to);
        }
    }
};
