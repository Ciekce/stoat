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
#include <span>
#include <string>
#include <string_view>

#include "bitboard.h"
#include "move.h"
#include "util/result.h"

namespace stoat {
    class Hand {
    public:
        Hand() = default;

        [[nodiscard]] constexpr u32 raw() const {
            return m_hand;
        }

        [[nodiscard]] inline bool empty() const {
            return m_hand == 0;
        }

        [[nodiscard]] u32 count(PieceType pt) const;

        u32 increment(PieceType pt);
        u32 decrement(PieceType pt);

        void set(PieceType pt, u32 count);

        [[nodiscard]] static Hand fromRaw(u32 raw);

        [[nodiscard]] std::string sfen(bool uppercase) const;

        [[nodiscard]] bool operator==(const Hand&) const = default;

    private:
        explicit Hand(u32 hand) :
                m_hand{hand} {}

        u32 m_hand{};
    };

    class SfenError {
    public:
        explicit SfenError(std::string message) :
                m_message{std::move(message)} {}

        [[nodiscard]] std::string_view message() const {
            return m_message;
        }

    private:
        std::string m_message{};
    };

    struct PositionKeys {
        u64 all{};

        void clear();

        void flipPiece(Piece piece, Square sq);
        void movePiece(Piece piece, Square from, Square to);

        void flipStm();

        void flipHandCount(Color c, PieceType pt, u32 count);
        void switchHandCount(Color c, PieceType pt, u32 before, u32 after);

        [[nodiscard]] bool operator==(const PositionKeys&) const = default;
    };

    enum class NnueUpdateAction {
        kNone = 0,
        kPush,
        kApplyInPlace,
    };

    enum class SennichiteStatus {
        kNone = 0,
        kDraw,
        kWin,  // perpetual check by opponent
        kLose, // perpetual check
    };

    namespace eval::nnue {
        struct NnueUpdates;
        class NnueState;
    } // namespace eval::nnue

    struct KingPair {
        std::array<Square, 2> squares{};

        [[nodiscard]] constexpr Square kingSq(Color c) const {
            assert(c);
            return squares[c.idx()];
        }

        [[nodiscard]] constexpr Square relativeKingSq(Color c) const {
            assert(c);
            return squares[c.idx()].relative(c);
        }

        [[nodiscard]] constexpr bool operator==(const KingPair& other) const = default;
    };

    class Position {
    public:
        Position();

        Position(const Position&) = default;
        Position(Position&&) = default;

        template <NnueUpdateAction kUpdateAction = NnueUpdateAction::kNone>
        [[nodiscard]] Position applyMove(Move move, eval::nnue::NnueState* nnueState = nullptr) const;

        [[nodiscard]] Position applyNullMove() const;

        [[nodiscard]] inline Bitboard occupancy() const {
            return m_colors[0] | m_colors[1];
        }

        [[nodiscard]] inline Bitboard colorBb(Color color) const {
            assert(color);
            return m_colors[color.idx()];
        }

        [[nodiscard]] inline Bitboard pieceTypeBb(PieceType pt) const {
            assert(pt);
            return m_pieces[pt.idx()];
        }

        [[nodiscard]] inline Bitboard pieceBb(Piece piece) const {
            assert(piece);
            return m_colors[piece.color().idx()] & m_pieces[piece.type().idx()];
        }

        [[nodiscard]] inline Bitboard pieceBb(PieceType pt, Color c) const {
            assert(pt);
            assert(c);
            return m_colors[c.idx()] & m_pieces[pt.idx()];
        }

        [[nodiscard]] inline Piece pieceOn(Square square) const {
            assert(square);
            return m_mailbox[square.idx()];
        }

        [[nodiscard]] inline const Hand& hand(Color color) const {
            assert(color);
            return m_hands[color.idx()];
        }

        [[nodiscard]] inline u64 key() const {
            return m_keys.all;
        }

        [[nodiscard]] u64 keyAfter(Move move) const;

        [[nodiscard]] inline bool isInCheck() const {
            return !m_checkers.empty();
        }

        [[nodiscard]] inline Bitboard checkers() const {
            return m_checkers;
        }

        [[nodiscard]] inline Bitboard pinned() const {
            return m_pinned;
        }

        [[nodiscard]] inline Color stm() const {
            return m_stm;
        }

        [[nodiscard]] inline u32 moveCount() const {
            return m_moveCount;
        }

        [[nodiscard]] inline Square kingSq(Color c) const {
            assert(c);
            return m_kingSquares.kingSq(c);
        }

        [[nodiscard]] inline KingPair kingSquares() const {
            return m_kingSquares;
        }

        [[nodiscard]] SennichiteStatus testSennichite(
            bool cuteChessWorkaround,
            std::span<const u64> keyHistory,
            i32 limit = 16
        ) const;

        [[nodiscard]] bool isEnteringKingsWin() const;

        [[nodiscard]] bool isPseudolegal(Move move) const;
        [[nodiscard]] bool isLegal(Move move) const;

        [[nodiscard]] bool isCapture(Move move) const;

        [[nodiscard]] bool isAttacked(Square sq, Color attacker, Bitboard occ) const;

        [[nodiscard]] bool isAttacked(Square sq, Color attacker) const {
            return isAttacked(sq, attacker, occupancy());
        }

        [[nodiscard]] Bitboard attackersTo(Square sq, Color attacker) const;

        [[nodiscard]] Bitboard allAttackersTo(Square sq, Bitboard occ) const;

        [[nodiscard]] std::string sfen() const;

        void regenKey();

        [[nodiscard]] bool operator==(const Position&) const = default;

        Position& operator=(const Position&) = default;
        Position& operator=(Position&&) = default;

        [[nodiscard]] static Position startpos();

        [[nodiscard]] static util::Result<Position, SfenError> fromSfenParts(std::span<std::string_view> sfen);
        [[nodiscard]] static util::Result<Position, SfenError> fromSfen(std::string_view sfen);

        friend std::ostream& operator<<(std::ostream& stream, const Position& pos);

    private:
        std::array<Bitboard, Colors::kCount> m_colors{};
        std::array<Bitboard, PieceTypes::kCount> m_pieces{};

        std::array<Piece, Squares::kCount> m_mailbox{};

        std::array<Hand, Colors::kCount> m_hands{};

        std::array<u16, Colors::kCount> m_consecutiveChecks{};

        PositionKeys m_keys{};

        Bitboard m_checkers{};
        Bitboard m_pinned{};

        KingPair m_kingSquares{};

        Color m_stm{Colors::kBlack};

        u16 m_moveCount{1};

        void addPiece(Square sq, Piece piece);

        template <bool kUpdateNnue>
        void movePiece(Square from, Square to, Piece piece, eval::nnue::NnueUpdates& nnueUpdates);
        template <bool kUpdateNnue>
        void promotePiece(Square from, Square to, Piece piece, eval::nnue::NnueUpdates& nnueUpdates);
        template <bool kUpdateNnue>
        void dropPiece(Square sq, Piece piece, eval::nnue::NnueUpdates& nnueUpdates);

        void updateAttacks();

        void regen();
    };
} // namespace stoat

template <>
struct fmt::formatter<stoat::Position> : fmt::formatter<std::string_view> {
    format_context::iterator format(const stoat::Position& value, format_context& ctx) const;
};

template <>
struct fmt::formatter<stoat::Hand> : fmt::formatter<std::string_view> {
    format_context::iterator format(const stoat::Hand& value, format_context& ctx) const;
};
