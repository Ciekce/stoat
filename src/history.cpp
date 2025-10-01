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

#include "history.h"

#include <cstring>

namespace stoat {
    namespace {
        inline void updateConthist(
            std::span<ContinuationSubtable*> continuations,
            i32 ply,
            const Position& pos,
            Move move,
            HistoryScore bonus,
            i32 offset
        ) {
            if (offset > ply) {
                return;
            }

            if (auto* ptr = continuations[ply - offset]) {
                (*ptr)[{pos, move}].update(bonus);
            }
        }

        inline HistoryScore conthistScore(
            std::span<ContinuationSubtable* const> continuations,
            i32 ply,
            const Position& pos,
            Move move,
            i32 offset
        ) {
            if (offset > ply) {
                return 0;
            }

            if (const auto* ptr = continuations[ply - offset]) {
                return (*ptr)[{pos, move}];
            }

            return 0;
        }
    } // namespace

    void HistoryTables::clear() {
        std::memset(m_nonCaptureNonDrop.data(), 0, sizeof(m_nonCaptureNonDrop));
        std::memset(m_drop.data(), 0, sizeof(m_drop));
        std::memset(m_continuation.data(), 0, sizeof(m_continuation));
        std::memset(m_capture.data(), 0, sizeof(m_capture));
    }

    i32 HistoryTables::mainNonCaptureScore(Move move) const {
        if (move.isDrop()) {
            return m_drop[move.dropPiece().idx()][move.to().idx()];
        } else {
            return m_nonCaptureNonDrop[move.isPromo()][move.from().idx()][move.to().idx()];
        }
    }

    i32 HistoryTables::nonCaptureScore(
        std::span<ContinuationSubtable* const> continuations,
        i32 ply,
        const Position& pos,
        Move move
    ) const {
        i32 score{};

        if (move.isDrop()) {
            score += m_drop[move.dropPiece().idx()][move.to().idx()];
        } else {
            score += m_nonCaptureNonDrop[move.isPromo()][move.from().idx()][move.to().idx()];
        }

        score += conthistScore(continuations, ply, pos, move, 1);
        score += conthistScore(continuations, ply, pos, move, 2);
        score += conthistScore(continuations, ply, pos, move, 3);

        return score;
    }

    void HistoryTables::updateNonCaptureScore(
        std::span<ContinuationSubtable*> continuations,
        i32 ply,
        const Position& pos,
        Move move,
        HistoryScore bonus
    ) {
        if (move.isDrop()) {
            m_drop[move.dropPiece().idx()][move.to().idx()].update(bonus);
        } else {
            m_nonCaptureNonDrop[move.isPromo()][move.from().idx()][move.to().idx()].update(bonus);
        }

        updateNonCaptureConthistScore(continuations, ply, pos, move, bonus);
    }

    void HistoryTables::updateNonCaptureConthistScore(
        std::span<ContinuationSubtable*> continuations,
        i32 ply,
        const Position& pos,
        Move move,
        HistoryScore bonus
    ) {
        updateConthist(continuations, ply, pos, move, bonus, 1);
        updateConthist(continuations, ply, pos, move, bonus, 2);
        updateConthist(continuations, ply, pos, move, bonus, 3);
    }

    i32 HistoryTables::captureScore(Move move, PieceType captured) const {
        return m_capture[move.isPromo()][move.from().idx()][move.to().idx()][captured.idx()];
    }

    void HistoryTables::updateCaptureScore(Move move, PieceType captured, HistoryScore bonus) {
        m_capture[move.isPromo()][move.from().idx()][move.to().idx()][captured.idx()].update(bonus);
    }
} // namespace stoat
