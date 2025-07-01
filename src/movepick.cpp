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

#include "movepick.h"

#include "see.h"

namespace stoat {
    Move MoveGenerator::next() {
        switch (m_stage) {
            case MovegenStage::kTtMove: {
                ++m_stage;

                if (m_ttMove && m_pos.isPseudolegal(m_ttMove)) {
                    return m_ttMove;
                }

                [[fallthrough]];
            }

            case MovegenStage::kGenerateCaptures: {
                movegen::generateCaptures(m_moves, m_pos);
                m_end = m_moves.size();

                scoreCaptures();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kGoodCaptures: {
                while (m_idx < m_end) {
                    const auto idx = findNext();
                    const auto move = m_moves[idx];

                    if (move == m_ttMove) {
                        continue;
                    }

                    if (see::see(m_pos, move, 0)) {
                        return move;
                    }

                    m_moves[m_badCapturesEnd++] = m_moves[idx];
                }

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kGenerateNonCaptures: {
                if (!m_skipNonCaptures) {
                    movegen::generateNonCaptures(m_moves, m_pos);
                    m_end = m_moves.size();
                }

                scoreNonCaptures();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kNonCaptures: {
                if (!m_skipNonCaptures) {
                    if (const auto move = selectNext([this](Move move) { return move != m_ttMove; })) {
                        return move;
                    }
                }

                m_idx = 0;
                m_end = m_badCapturesEnd;

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kBadCaptures: {
                if (const auto move = selectNext<false>([this](auto move) { return move != m_ttMove; })) {
                    return move;
                }

                m_stage = MovegenStage::kEnd;
                return kNullMove;
            }

            case MovegenStage::kQsearchGenerateCaptures: {
                movegen::generateCaptures(m_moves, m_pos);
                m_end = m_moves.size();

                scoreCaptures();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kQsearchCaptures: {
                if (const auto move = selectNext([](Move) { return true; })) {
                    return move;
                }

                m_stage = MovegenStage::kEnd;
                return kNullMove;
            }

            case MovegenStage::kQsearchEvasionsGenerateCaptures: {
                movegen::generateCaptures(m_moves, m_pos);
                m_end = m_moves.size();

                scoreCaptures();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kQsearchEvasionsCaptures: {
                if (const auto move = selectNext([](Move) { return true; })) {
                    return move;
                }

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kQsearchEvasionsGenerateNonCaptures: {
                if (!m_skipNonCaptures) {
                    movegen::generateNonCaptures(m_moves, m_pos);
                    m_end = m_moves.size();
                }

                scoreNonCaptures();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kQsearchEvasionsNonCaptures: {
                if (!m_skipNonCaptures) {
                    if (const auto move = selectNext([this](Move move) { return move != m_ttMove; })) {
                        return move;
                    }
                }

                m_stage = MovegenStage::kEnd;
                return kNullMove;
            }

            default:
                return kNullMove;
        }
    }

    MoveGenerator MoveGenerator::main(
        const Position& pos,
        Move ttMove,
        const HistoryTables& history,
        std::span<ContinuationSubtable* const> continuations,
        i32 ply
    ) {
        assert(continuations.size() == kMaxDepth + 1);
        return MoveGenerator{MovegenStage::kTtMove, pos, ttMove, history, continuations, ply};
    }

    MoveGenerator MoveGenerator::qsearch(
        const Position& pos,
        const HistoryTables& history,
        std::span<ContinuationSubtable* const> continuations,
        i32 ply
    ) {
        assert(continuations.size() == kMaxDepth + 1);
        const auto initialStage =
            pos.isInCheck() ? MovegenStage::kQsearchEvasionsGenerateCaptures : MovegenStage::kQsearchGenerateCaptures;
        return MoveGenerator{initialStage, pos, kNullMove, history, continuations, ply};
    }

    MoveGenerator::MoveGenerator(
        MovegenStage initialStage,
        const Position& pos,
        Move ttMove,
        const HistoryTables& history,
        std::span<ContinuationSubtable* const> continuations,
        i32 ply
    ) :
            m_stage{initialStage},
            m_pos{pos},
            m_ttMove{ttMove},
            m_history{history},
            m_continuations{continuations},
            m_ply{ply} {}

    i32 MoveGenerator::scoreCapture(Move move) {
        const auto captured = m_pos.pieceOn(move.to()).type();
        return see::pieceValue(captured) + m_history.captureScore(move, captured) / 8;
    }

    void MoveGenerator::scoreCaptures() {
        for (usize idx = m_idx; idx < m_end; ++idx) {
            m_scores[idx] = scoreCapture(m_moves[idx]);
        }
    }

    i32 MoveGenerator::scoreNonCapture(Move move) {
        return m_history.nonCaptureScore(m_continuations, m_ply, m_pos, move);
    }

    void MoveGenerator::scoreNonCaptures() {
        for (usize idx = m_idx; idx < m_end; ++idx) {
            m_scores[idx] = scoreNonCapture(m_moves[idx]);
        }
    }

    usize MoveGenerator::findNext() {
        auto bestIdx = m_idx;
        auto bestScore = m_scores[m_idx];

        for (usize idx = m_idx + 1; idx < m_end; ++idx) {
            if (m_scores[idx] > bestScore) {
                bestIdx = idx;
                bestScore = m_scores[idx];
            }
        }

        if (bestIdx != m_idx) {
            std::swap(m_moves[m_idx], m_moves[bestIdx]);
            std::swap(m_scores[m_idx], m_scores[bestIdx]);
        }

        return m_idx++;
    }
} // namespace stoat
