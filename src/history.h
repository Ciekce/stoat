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

#include <algorithm>
#include <utility>

#include "core.h"
#include "move.h"
#include "position.h"
#include "util/multi_array.h"

namespace stoat {
    using HistoryScore = i16;

    struct HistoryEntry {
        i16 value{};

        HistoryEntry() = default;
        HistoryEntry(HistoryScore v) :
                value{v} {}

        [[nodiscard]] inline operator HistoryScore() const {
            return value;
        }

        [[nodiscard]] inline HistoryEntry& operator=(HistoryScore v) {
            value = v;
            return *this;
        }

        inline void update(HistoryScore bonus) {
            value += bonus - value * std::abs(bonus) / 16384;
        }
    };

    class ContinuationSubtable {
    public:
        //TODO take two args when c++23 is usable
        inline HistoryScore operator[](std::pair<const Position&, Move> ctx) const {
            const auto [pos, move] = ctx;
            if (move.isDrop()) {
                return m_data[true][move.dropPiece().withColor(pos.stm()).idx()][move.to().idx()];
            } else {
                return m_data[false][pos.pieceOn(move.from()).idx()][move.to().idx()];
            }
        }

        inline HistoryEntry& operator[](std::pair<const Position&, Move> ctx) {
            const auto [pos, move] = ctx;
            if (move.isDrop()) {
                return m_data[true][move.dropPiece().withColor(pos.stm()).idx()][move.to().idx()];
            } else {
                return m_data[false][pos.pieceOn(move.from()).idx()][move.to().idx()];
            }
        }

    private:
        // [drop][piece][to]
        util::MultiArray<HistoryEntry, 2, Pieces::kCount, Squares::kCount> m_data{};
    };

    [[nodiscard]] constexpr HistoryScore historyBonus(i32 depth, f64 complexityFactor) {
        return static_cast<HistoryScore>(std::clamp(static_cast<i32>((depth * 300 - 300) * complexityFactor), 0, 2500));
    }

    class HistoryTables {
    public:
        void clear();

        [[nodiscard]] inline const ContinuationSubtable& contTable(const Position& pos, Move move) const {
            if (move.isDrop()) {
                return m_continuation[true][move.dropPiece().withColor(pos.stm()).idx()][move.to().idx()];
            } else {
                return m_continuation[false][pos.pieceOn(move.from()).idx()][move.to().idx()];
            }
        }

        [[nodiscard]] inline ContinuationSubtable& contTable(const Position& pos, Move move) {
            if (move.isDrop()) {
                return m_continuation[true][move.dropPiece().withColor(pos.stm()).idx()][move.to().idx()];
            } else {
                return m_continuation[false][pos.pieceOn(move.from()).idx()][move.to().idx()];
            }
        }

        [[nodiscard]] i32 mainNonCaptureScore(Move move) const;

        [[nodiscard]] i32 nonCaptureScore(
            std::span<ContinuationSubtable* const> continuations,
            i32 ply,
            const Position& pos,
            Move move
        ) const;

        void updateNonCaptureScore(
            std::span<ContinuationSubtable*> continuations,
            i32 ply,
            const Position& pos,
            Move move,
            HistoryScore bonus
        );

        void updateNonCaptureConthistScore(
            std::span<ContinuationSubtable*> continuations,
            i32 ply,
            const Position& pos,
            Move move,
            HistoryScore bonus
        );

        [[nodiscard]] i32 captureScore(Move move, PieceType captured) const;
        void updateCaptureScore(Move move, PieceType captured, HistoryScore bonus);

    private:
        // [promo][from][to]
        util::MultiArray<HistoryEntry, 2, Squares::kCount, Squares::kCount> m_nonCaptureNonDrop{};
        // [dropped piece type][drop square]
        util::MultiArray<HistoryEntry, PieceTypes::kCount, Squares::kCount> m_drop{};

        // [drop][prev piece][to]
        util::MultiArray<ContinuationSubtable, 2, Pieces::kCount, Squares::kCount> m_continuation{};

        // [promo][from][to][captured]
        util::MultiArray<HistoryEntry, 2, Squares::kCount, Squares::kCount, PieceTypes::kCount> m_capture{};
    };
} // namespace stoat
