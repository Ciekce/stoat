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
#include <bit>
#include <cassert>

namespace stoat {
    class Color {
    public:
        constexpr Color() = default;

        constexpr Color(const Color&) = default;
        constexpr Color(Color&&) = default;

        [[nodiscard]] constexpr u8 raw() const {
            return m_id;
        }

        [[nodiscard]] constexpr usize idx() const {
            return static_cast<usize>(m_id);
        }

        [[nodiscard]] constexpr Color flip() const {
            assert(m_id != kNoneId);
            return Color{static_cast<u8>(m_id ^ 1)};
        }

        [[nodiscard]] static constexpr Color fromRaw(u8 id) {
            assert(id <= kNoneId);
            return Color{id};
        }

        [[nodiscard]] constexpr explicit operator bool() const {
            return m_id != kNoneId;
        }

        [[nodiscard]] constexpr bool operator==(const Color&) const = default;

        constexpr Color& operator=(const Color&) = default;
        constexpr Color& operator=(Color&&) = default;

    private:
        explicit constexpr Color(u8 id) :
                m_id{id} {}

        u8 m_id{};

        enum : u8 {
            kBlackId = 0,
            kWhiteId,
            kNoneId,
        };

        friend struct Colors;
    };

    struct Colors {
        Colors() = delete;

        static constexpr Color kBlack{Color::kBlackId};
        static constexpr Color kWhite{Color::kWhiteId};
        static constexpr Color kNone{Color::kNoneId};

        static constexpr usize kCount = kNone.idx();
    };

    class Piece;

    class PieceType {
    public:
        constexpr PieceType() = default;

        constexpr PieceType(const PieceType&) = default;
        constexpr PieceType(PieceType&&) = default;

        [[nodiscard]] constexpr u8 raw() const {
            return m_id;
        }

        [[nodiscard]] constexpr usize idx() const {
            return static_cast<usize>(m_id);
        }

        [[nodiscard]] constexpr bool isPromoted() const {
            assert(m_id <= kNoneId);
            return m_id == kPromotedPawnId || m_id == kPromotedLanceId || m_id == kPromotedKnightId
                || m_id == kPromotedSilverId || m_id == kPromotedBishopId || m_id == kPromotedRookId;
        }

        [[nodiscard]] constexpr bool canPromote() const {
            assert(m_id <= kNoneId);
            return m_id == kPawnId || m_id == kLanceId || m_id == kKnightId || m_id == kSilverId || m_id == kBishopId
                || m_id == kRookId;
        }

        [[nodiscard]] constexpr Piece withColor(Color c) const;

        [[nodiscard]] constexpr PieceType promoted() const {
            switch (m_id) {
                case kPawnId:
                    return PieceType{kPromotedPawnId};
                case kLanceId:
                    return PieceType{kPromotedLanceId};
                case kKnightId:
                    return PieceType{kPromotedKnightId};
                case kSilverId:
                    return PieceType{kPromotedSilverId};
                case kBishopId:
                    return PieceType{kPromotedBishopId};
                case kRookId:
                    return PieceType{kPromotedRookId};
                default:
                    return PieceType{kNoneId};
            }
        }

        [[nodiscard]] constexpr PieceType unpromoted() const {
            switch (m_id) {
                case kPromotedPawnId:
                    return PieceType{kPawnId};
                case kPromotedLanceId:
                    return PieceType{kLanceId};
                case kPromotedKnightId:
                    return PieceType{kKnightId};
                case kPromotedSilverId:
                    return PieceType{kSilverId};
                case kPromotedBishopId:
                    return PieceType{kBishopId};
                case kPromotedRookId:
                    return PieceType{kRookId};
                default:
                    return *this;
            }
        }

        [[nodiscard]] constexpr std::string_view str() const {
            switch (m_id) {
                case kPawnId:
                    return "P";
                case kPromotedPawnId:
                    return "+P";
                case kLanceId:
                    return "L";
                case kKnightId:
                    return "N";
                case kPromotedLanceId:
                    return "+L";
                case kPromotedKnightId:
                    return "+N";
                case kSilverId:
                    return "S";
                case kPromotedSilverId:
                    return "+S";
                case kGoldId:
                    return "G";
                case kBishopId:
                    return "B";
                case kRookId:
                    return "R";
                case kPromotedBishopId:
                    return "+B";
                case kPromotedRookId:
                    return "+R";
                case kKingId:
                    return "K";
                default:
                    return "?";
            }
        }

        [[nodiscard]] static constexpr PieceType fromRaw(u8 id) {
            assert(id <= kNoneId);
            return PieceType{id};
        }

        [[nodiscard]] static constexpr PieceType unpromotedFromChar(char c) {
            switch (c) {
                case 'P':
                    return PieceType{kPawnId};
                case 'L':
                    return PieceType{kLanceId};
                case 'N':
                    return PieceType{kKnightId};
                case 'S':
                    return PieceType{kSilverId};
                case 'G':
                    return PieceType{kGoldId};
                case 'B':
                    return PieceType{kBishopId};
                case 'R':
                    return PieceType{kRookId};
                case 'K':
                    return PieceType{kKingId};
                default:
                    return PieceType{kNoneId};
            }
        }

        [[nodiscard]] constexpr explicit operator bool() const {
            return m_id != kNoneId;
        }

        [[nodiscard]] constexpr bool operator==(const PieceType&) const = default;

        constexpr PieceType& operator=(const PieceType&) = default;
        constexpr PieceType& operator=(PieceType&&) = default;

    private:
        explicit constexpr PieceType(u8 id) :
                m_id{id} {}

        u8 m_id{};

        enum : u8 {
            kPawnId = 0,
            kPromotedPawnId,
            kLanceId,
            kKnightId,
            kPromotedLanceId,
            kPromotedKnightId,
            kSilverId,
            kPromotedSilverId,
            kGoldId,
            kBishopId,
            kRookId,
            kPromotedBishopId,
            kPromotedRookId,
            kKingId,
            kNoneId,
        };

        friend struct PieceTypes;
    };

    struct PieceTypes {
        PieceTypes() = delete;

        static constexpr PieceType kPawn{PieceType::kPawnId};
        static constexpr PieceType kPromotedPawn{PieceType::kPromotedPawnId};
        static constexpr PieceType kLance{PieceType::kLanceId};
        static constexpr PieceType kKnight{PieceType::kKnightId};
        static constexpr PieceType kPromotedLance{PieceType::kPromotedLanceId};
        static constexpr PieceType kPromotedKnight{PieceType::kPromotedKnightId};
        static constexpr PieceType kSilver{PieceType::kSilverId};
        static constexpr PieceType kPromotedSilver{PieceType::kPromotedSilverId};
        static constexpr PieceType kGold{PieceType::kGoldId};
        static constexpr PieceType kBishop{PieceType::kBishopId};
        static constexpr PieceType kRook{PieceType::kRookId};
        static constexpr PieceType kPromotedBishop{PieceType::kPromotedBishopId};
        static constexpr PieceType kPromotedRook{PieceType::kPromotedRookId};
        static constexpr PieceType kKing{PieceType::kKingId};
        static constexpr PieceType kNone{PieceType::kNoneId};

        static constexpr usize kCount = kNone.idx();

        static constexpr std::array kAll = {
            kPawn,
            kPromotedPawn,
            kLance,
            kKnight,
            kPromotedLance,
            kPromotedKnight,
            kSilver,
            kPromotedSilver,
            kGold,
            kBishop,
            kRook,
            kPromotedBishop,
            kPromotedRook,
            kKing,
        };
    };

    class Piece {
    public:
        constexpr Piece() = default;

        constexpr Piece(const Piece&) = default;
        constexpr Piece(Piece&&) = default;

        [[nodiscard]] constexpr u8 raw() const {
            return m_id;
        }

        [[nodiscard]] constexpr usize idx() const {
            return static_cast<usize>(m_id);
        }

        [[nodiscard]] constexpr bool isPromoted() const {
            assert(m_id <= kNoneId);
            return type().isPromoted();
        }

        [[nodiscard]] constexpr PieceType type() const {
            assert(m_id != kNoneId);
            return PieceType::fromRaw(m_id >> 1);
        }

        [[nodiscard]] constexpr PieceType typeOrNone() const {
            return PieceType::fromRaw(m_id >> 1);
        }

        [[nodiscard]] constexpr Color color() const {
            assert(m_id != kNoneId);
            return Color::fromRaw(m_id & 0b1);
        }

        [[nodiscard]] constexpr Piece promoted() const {
            return type().promoted().withColor(color());
        }

        [[nodiscard]] constexpr std::string_view str() const {
            switch (m_id) {
                case kBlackPawnId:
                    return "P";
                case kWhitePawnId:
                    return "p";
                case kBlackPromotedPawnId:
                    return "+P";
                case kWhitePromotedPawnId:
                    return "+p";
                case kBlackLanceId:
                    return "L";
                case kWhiteLanceId:
                    return "l";
                case kBlackKnightId:
                    return "N";
                case kWhiteKnightId:
                    return "n";
                case kBlackPromotedLanceId:
                    return "+L";
                case kWhitePromotedLanceId:
                    return "+l";
                case kBlackPromotedKnightId:
                    return "+N";
                case kWhitePromotedKnightId:
                    return "+n";
                case kBlackSilverId:
                    return "S";
                case kWhiteSilverId:
                    return "s";
                case kBlackPromotedSilverId:
                    return "+S";
                case kWhitePromotedSilverId:
                    return "+s";
                case kBlackGoldId:
                    return "G";
                case kWhiteGoldId:
                    return "g";
                case kBlackBishopId:
                    return "B";
                case kWhiteBishopId:
                    return "b";
                case kBlackRookId:
                    return "R";
                case kWhiteRookId:
                    return "r";
                case kBlackPromotedBishopId:
                    return "+B";
                case kWhitePromotedBishopId:
                    return "+b";
                case kBlackPromotedRookId:
                    return "+R";
                case kWhitePromotedRookId:
                    return "+r";
                case kBlackKingId:
                    return "K";
                case kWhiteKingId:
                    return "k";
                default:
                    return "?";
            }
        }

        [[nodiscard]] static constexpr Piece fromRaw(u8 id) {
            assert(id <= kNoneId);
            return Piece{id};
        }

        [[nodiscard]] static constexpr Piece unpromotedFromChar(char c) {
            switch (c) {
                case 'P':
                    return Piece{kBlackPawnId};
                case 'p':
                    return Piece{kWhitePawnId};
                case 'L':
                    return Piece{kBlackLanceId};
                case 'l':
                    return Piece{kWhiteLanceId};
                case 'N':
                    return Piece{kBlackKnightId};
                case 'n':
                    return Piece{kWhiteKnightId};
                case 'S':
                    return Piece{kBlackSilverId};
                case 's':
                    return Piece{kWhiteSilverId};
                case 'G':
                    return Piece{kBlackGoldId};
                case 'g':
                    return Piece{kWhiteGoldId};
                case 'B':
                    return Piece{kBlackBishopId};
                case 'b':
                    return Piece{kWhiteBishopId};
                case 'R':
                    return Piece{kBlackRookId};
                case 'r':
                    return Piece{kWhiteRookId};
                case 'K':
                    return Piece{kBlackKingId};
                case 'k':
                    return Piece{kWhiteKingId};
                default:
                    return Piece{kNoneId};
            }
        }

        [[nodiscard]] static constexpr Piece fromStr(std::string_view str) {
            if (str.empty()) {
                return Piece{kNoneId};
            }

            if (str.length() == 2 && str[0] == ' ') {
                str = str.substr(1);
            }

            if ((str.length() != 1 && str.length() != 2) || (str.length() == 2 && str[0] != '+')) {
                return Piece{kNoneId};
            }

            const auto piece = unpromotedFromChar(str.back());

            if (str.length() == 2 && !piece.type().canPromote()) {
                return Piece{kNoneId};
            }

            return str.length() == 2 ? piece.promoted() : piece;
        }

        [[nodiscard]] constexpr explicit operator bool() const {
            return m_id != kNoneId;
        }

        [[nodiscard]] constexpr bool operator==(const Piece&) const = default;

        constexpr Piece& operator=(const Piece&) = default;
        constexpr Piece& operator=(Piece&&) = default;

    private:
        explicit constexpr Piece(u8 id) :
                m_id{id} {}

        u8 m_id{};

        enum : u8 {
            kBlackPawnId = 0,
            kWhitePawnId,
            kBlackPromotedPawnId,
            kWhitePromotedPawnId,
            kBlackLanceId,
            kWhiteLanceId,
            kBlackKnightId,
            kWhiteKnightId,
            kBlackPromotedLanceId,
            kWhitePromotedLanceId,
            kBlackPromotedKnightId,
            kWhitePromotedKnightId,
            kBlackSilverId,
            kWhiteSilverId,
            kBlackPromotedSilverId,
            kWhitePromotedSilverId,
            kBlackGoldId,
            kWhiteGoldId,
            kBlackBishopId,
            kWhiteBishopId,
            kBlackRookId,
            kWhiteRookId,
            kBlackPromotedBishopId,
            kWhitePromotedBishopId,
            kBlackPromotedRookId,
            kWhitePromotedRookId,
            kBlackKingId,
            kWhiteKingId,
            kNoneId,
        };

        friend struct Pieces;
    };

    constexpr Piece PieceType::withColor(Color c) const {
        assert(*this != PieceTypes::kNone);
        assert(c != Colors::kNone);

        return Piece::fromRaw((m_id << 1) | c.raw());
    }

    struct Pieces {
        Pieces() = delete;

        static constexpr Piece kBlackPawn{Piece::kBlackPawnId};
        static constexpr Piece kWhitePawn{Piece::kWhitePawnId};
        static constexpr Piece kBlackPromotedPawn{Piece::kBlackPromotedPawnId};
        static constexpr Piece kWhitePromotedPawn{Piece::kWhitePromotedPawnId};
        static constexpr Piece kBlackLance{Piece::kBlackLanceId};
        static constexpr Piece kWhiteLance{Piece::kWhiteLanceId};
        static constexpr Piece kBlackKnight{Piece::kBlackKnightId};
        static constexpr Piece kWhiteKnight{Piece::kWhiteKnightId};
        static constexpr Piece kBlackPromotedLance{Piece::kBlackPromotedLanceId};
        static constexpr Piece kWhitePromotedLance{Piece::kWhitePromotedLanceId};
        static constexpr Piece kBlackPromotedKnight{Piece::kBlackPromotedKnightId};
        static constexpr Piece kWhitePromotedKnight{Piece::kWhitePromotedKnightId};
        static constexpr Piece kBlackSilver{Piece::kBlackSilverId};
        static constexpr Piece kWhiteSilver{Piece::kWhiteSilverId};
        static constexpr Piece kBlackPromotedSilver{Piece::kBlackPromotedSilverId};
        static constexpr Piece kWhitePromotedSilver{Piece::kWhitePromotedSilverId};
        static constexpr Piece kBlackGold{Piece::kBlackGoldId};
        static constexpr Piece kWhiteGold{Piece::kWhiteGoldId};
        static constexpr Piece kBlackBishop{Piece::kBlackBishopId};
        static constexpr Piece kWhiteBishop{Piece::kWhiteBishopId};
        static constexpr Piece kBlackRook{Piece::kBlackRookId};
        static constexpr Piece kWhiteRook{Piece::kWhiteRookId};
        static constexpr Piece kBlackPromotedBishop{Piece::kBlackPromotedBishopId};
        static constexpr Piece kWhitePromotedBishop{Piece::kWhitePromotedBishopId};
        static constexpr Piece kBlackPromotedRook{Piece::kBlackPromotedRookId};
        static constexpr Piece kWhitePromotedRook{Piece::kWhitePromotedRookId};
        static constexpr Piece kBlackKing{Piece::kBlackKingId};
        static constexpr Piece kWhiteKing{Piece::kWhiteKingId};
        static constexpr Piece kNone{Piece::kNoneId};

        static constexpr usize kCount = kNone.idx();
    };

    class Square {
    public:
        constexpr Square() = default;

        constexpr Square(const Square&) = default;
        constexpr Square(Square&&) = default;

        [[nodiscard]] constexpr u8 raw() const {
            return m_id;
        }

        [[nodiscard]] constexpr usize idx() const {
            return static_cast<usize>(m_id);
        }

        [[nodiscard]] constexpr i32 rank() const {
            assert(m_id != kNoneId);
            return static_cast<i32>(m_id) / 9;
        }

        [[nodiscard]] constexpr i32 file() const {
            assert(m_id != kNoneId);
            return static_cast<i32>(m_id) % 9;
        }

        [[nodiscard]] constexpr Square rotate() const {
            assert(m_id != kNoneId);
            return fromRaw(80 - m_id);
        }

        [[nodiscard]] constexpr Square flipFile() const {
            assert(m_id != kNoneId);
            return fromFileRank(8 - file(), rank());
        }

        [[nodiscard]] constexpr Square relative(Color c) const {
            assert(m_id != kNoneId);
            if (c == Colors::kBlack) {
                return *this;
            } else {
                return rotate();
            }
        }

        [[nodiscard]] constexpr u128 bit() const {
            assert(m_id != kNoneId);
            return u128{1} << m_id;
        }

        [[nodiscard]] constexpr Square offset(i32 offset) const {
            assert(m_id + offset >= 0);
            assert(m_id + offset < kNoneId);
            return fromRaw(m_id + offset);
        }

        [[nodiscard]] static i32 chebyshev(Square a, Square b) {
            return std::max(
                std::abs(static_cast<i32>(a.rank()) - static_cast<i32>(b.rank())),
                std::abs(static_cast<i32>(a.file()) - static_cast<i32>(b.file()))
            );
        }

        [[nodiscard]] static constexpr Square fromRaw(u8 id) {
            assert(id <= kNoneId);
            return Square{id};
        }

        [[nodiscard]] static constexpr Square fromFileRank(i32 file, i32 rank) {
            assert(rank >= 0 && rank <= 8);
            assert(file >= 0 && file <= 8);
            return fromRaw(rank * 9 + file);
        }

        [[nodiscard]] static constexpr Square fromStr(std::string_view str) {
            if (str.length() != 2) {
                return Square{kNoneId};
            }

            if (str[0] < '1' || str[0] > '9' || str[1] < 'a' || str[1] > 'i') {
                return Square{kNoneId};
            }

            const u32 file = '1' + 8 - str[0];
            const u32 rank = 'a' + 8 - str[1];

            return fromRaw(rank * 9 + file);
        }

        [[nodiscard]] constexpr explicit operator bool() const {
            return m_id != kNoneId;
        }

        [[nodiscard]] constexpr bool operator==(const Square&) const = default;

        constexpr Square& operator=(const Square&) = default;
        constexpr Square& operator=(Square&&) = default;

    private:
        explicit constexpr Square(u8 id) :
                m_id{id} {}

        u8 m_id{};

        enum : u8 {
            k9IId,
            k8IId,
            k7IId,
            k6IId,
            k5IId,
            k4IId,
            k3IId,
            k2IId,
            k1IId,
            k9HId,
            k8HId,
            k7HId,
            k6HId,
            k5HId,
            k4HId,
            k3HId,
            k2HId,
            k1HId,
            k9GId,
            k8GId,
            k7GId,
            k6GId,
            k5GId,
            k4GId,
            k3GId,
            k2GId,
            k1GId,
            k9FId,
            k8FId,
            k7FId,
            k6FId,
            k5FId,
            k4FId,
            k3FId,
            k2FId,
            k1FId,
            k9EId,
            k8EId,
            k7EId,
            k6EId,
            k5EId,
            k4EId,
            k3EId,
            k2EId,
            k1EId,
            k9DId,
            k8DId,
            k7DId,
            k6DId,
            k5DId,
            k4DId,
            k3DId,
            k2DId,
            k1DId,
            k9CId,
            k8CId,
            k7CId,
            k6CId,
            k5CId,
            k4CId,
            k3CId,
            k2CId,
            k1CId,
            k9BId,
            k8BId,
            k7BId,
            k6BId,
            k5BId,
            k4BId,
            k3BId,
            k2BId,
            k1BId,
            k9AId,
            k8AId,
            k7AId,
            k6AId,
            k5AId,
            k4AId,
            k3AId,
            k2AId,
            k1AId,
            kNoneId,
        };

        friend struct Squares;
    };

    struct Squares {
        Squares() = delete;

        static constexpr Square k9I{Square::k9IId};
        static constexpr Square k8I{Square::k8IId};
        static constexpr Square k7I{Square::k7IId};
        static constexpr Square k6I{Square::k6IId};
        static constexpr Square k5I{Square::k5IId};
        static constexpr Square k4I{Square::k4IId};
        static constexpr Square k3I{Square::k3IId};
        static constexpr Square k2I{Square::k2IId};
        static constexpr Square k1I{Square::k1IId};
        static constexpr Square k9H{Square::k9HId};
        static constexpr Square k8H{Square::k8HId};
        static constexpr Square k7H{Square::k7HId};
        static constexpr Square k6H{Square::k6HId};
        static constexpr Square k5H{Square::k5HId};
        static constexpr Square k4H{Square::k4HId};
        static constexpr Square k3H{Square::k3HId};
        static constexpr Square k2H{Square::k2HId};
        static constexpr Square k1H{Square::k1HId};
        static constexpr Square k9G{Square::k9GId};
        static constexpr Square k8G{Square::k8GId};
        static constexpr Square k7G{Square::k7GId};
        static constexpr Square k6G{Square::k6GId};
        static constexpr Square k5G{Square::k5GId};
        static constexpr Square k4G{Square::k4GId};
        static constexpr Square k3G{Square::k3GId};
        static constexpr Square k2G{Square::k2GId};
        static constexpr Square k1G{Square::k1GId};
        static constexpr Square k9F{Square::k9FId};
        static constexpr Square k8F{Square::k8FId};
        static constexpr Square k7F{Square::k7FId};
        static constexpr Square k6F{Square::k6FId};
        static constexpr Square k5F{Square::k5FId};
        static constexpr Square k4F{Square::k4FId};
        static constexpr Square k3F{Square::k3FId};
        static constexpr Square k2F{Square::k2FId};
        static constexpr Square k1F{Square::k1FId};
        static constexpr Square k9E{Square::k9EId};
        static constexpr Square k8E{Square::k8EId};
        static constexpr Square k7E{Square::k7EId};
        static constexpr Square k6E{Square::k6EId};
        static constexpr Square k5E{Square::k5EId};
        static constexpr Square k4E{Square::k4EId};
        static constexpr Square k3E{Square::k3EId};
        static constexpr Square k2E{Square::k2EId};
        static constexpr Square k1E{Square::k1EId};
        static constexpr Square k9D{Square::k9DId};
        static constexpr Square k8D{Square::k8DId};
        static constexpr Square k7D{Square::k7DId};
        static constexpr Square k6D{Square::k6DId};
        static constexpr Square k5D{Square::k5DId};
        static constexpr Square k4D{Square::k4DId};
        static constexpr Square k3D{Square::k3DId};
        static constexpr Square k2D{Square::k2DId};
        static constexpr Square k1D{Square::k1DId};
        static constexpr Square k9C{Square::k9CId};
        static constexpr Square k8C{Square::k8CId};
        static constexpr Square k7C{Square::k7CId};
        static constexpr Square k6C{Square::k6CId};
        static constexpr Square k5C{Square::k5CId};
        static constexpr Square k4C{Square::k4CId};
        static constexpr Square k3C{Square::k3CId};
        static constexpr Square k2C{Square::k2CId};
        static constexpr Square k1C{Square::k1CId};
        static constexpr Square k9B{Square::k9BId};
        static constexpr Square k8B{Square::k8BId};
        static constexpr Square k7B{Square::k7BId};
        static constexpr Square k6B{Square::k6BId};
        static constexpr Square k5B{Square::k5BId};
        static constexpr Square k4B{Square::k4BId};
        static constexpr Square k3B{Square::k3BId};
        static constexpr Square k2B{Square::k2BId};
        static constexpr Square k1B{Square::k1BId};
        static constexpr Square k9A{Square::k9AId};
        static constexpr Square k8A{Square::k8AId};
        static constexpr Square k7A{Square::k7AId};
        static constexpr Square k6A{Square::k6AId};
        static constexpr Square k5A{Square::k5AId};
        static constexpr Square k4A{Square::k4AId};
        static constexpr Square k3A{Square::k3AId};
        static constexpr Square k2A{Square::k2AId};
        static constexpr Square k1A{Square::k1AId};
        static constexpr Square kNone{Square::kNoneId};

        static constexpr usize kCount = kNone.idx();
    };

    constexpr auto kMaxInHand = [] {
        const auto round = [](u32 count) { return (std::bit_floor(count) << 1) - 1; };

        std::array<u32, PieceTypes::kCount> offsets{};

        offsets[PieceTypes::kPawn.idx()] = round(18);
        offsets[PieceTypes::kLance.idx()] = round(4);
        offsets[PieceTypes::kKnight.idx()] = round(4);
        offsets[PieceTypes::kSilver.idx()] = round(4);
        offsets[PieceTypes::kGold.idx()] = round(4);
        offsets[PieceTypes::kBishop.idx()] = round(2);
        offsets[PieceTypes::kRook.idx()] = round(2);

        return offsets;
    }();

    [[nodiscard]] constexpr u32 maxPiecesInHand(PieceType pt) {
        assert(pt);
        return kMaxInHand[pt.idx()];
    }

    constexpr std::array kHandPieces = {
        PieceTypes::kPawn,
        PieceTypes::kLance,
        PieceTypes::kKnight,
        PieceTypes::kSilver,
        PieceTypes::kGold,
        PieceTypes::kBishop,
        PieceTypes::kRook,
    };

    using Score = i32;

    constexpr auto kScoreInf = 32767;
    constexpr auto kScoreMate = 32766;
    constexpr auto kScoreWin = 25000;

    constexpr auto kScoreNone = -kScoreInf;

    constexpr i32 kMaxDepth = 255;

    constexpr auto kScoreMaxMate = kScoreMate - kMaxDepth;
} // namespace stoat

template <>
struct fmt::formatter<stoat::PieceType> : fmt::formatter<std::string_view> {
    constexpr format_context::iterator format(stoat::PieceType value, format_context& ctx) const {
        return formatter<std::string_view>::format(value.str(), ctx);
    }
};

template <>
struct fmt::formatter<stoat::Piece> : fmt::formatter<std::string_view> {
    constexpr format_context::iterator format(stoat::Piece value, format_context& ctx) const {
        return formatter<std::string_view>::format(value.str(), ctx);
    }
};

template <>
struct fmt::formatter<stoat::Square> : fmt::nested_formatter<char> {
    format_context::iterator format(stoat::Square value, format_context& ctx) const {
        using namespace stoat;

        return write_padded(ctx, [=, this](auto out) {
            if (value == Squares::kNone) {
                return format_to(out, "??");
            }

            return format_to(
                out,
                "{}{}",
                nested(static_cast<char>('1' + 8 - value.file())),
                nested(static_cast<char>('a' + 8 - value.rank()))
            );
        });
    }
};
