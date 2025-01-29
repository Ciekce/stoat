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

namespace stoat {
    Move MoveGenerator::next() {
        switch (m_stage) {
            case MovegenStage::TtMove: {
                ++m_stage;

                if (m_ttMove && m_pos.isPseudolegal(m_ttMove)) {
                    return m_ttMove;
                }

                [[fallthrough]];
            }

            case MovegenStage::GenerateCaptures: {
                movegen::generateCaptures(m_moves, m_pos);
                m_end = m_moves.size();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::Captures: {
                if (const auto move = selectNext([this](Move move) { return move != m_ttMove; })) {
                    return move;
                }

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::GenerateNonCaptures: {
                movegen::generateNonCaptures(m_moves, m_pos);
                m_end = m_moves.size();

                scoreNonCaptures();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::NonCaptures: {
                if (const auto move = selectNext([this](Move move) { return move != m_ttMove; })) {
                    return move;
                }

                m_stage = MovegenStage::End;
                return kNullMove;
            }

            case MovegenStage::QsearchGenerateCaptures: {
                movegen::generateCaptures(m_moves, m_pos);
                m_end = m_moves.size();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::QsearchCaptures: {
                if (const auto move = selectNext([](Move) { return true; })) {
                    return move;
                }

                m_stage = MovegenStage::End;
                return kNullMove;
            }

            case MovegenStage::QsearchGenerateRecaptures: {
                movegen::generateRecaptures(m_moves, m_pos, m_captureSq);
                m_end = m_moves.size();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::QsearchRecaptures: {
                if (const auto move = selectNext([](Move) { return true; })) {
                    return move;
                }

                m_stage = MovegenStage::End;
                return kNullMove;
            }

            default:
                return kNullMove;
        }
    }

    MoveGenerator MoveGenerator::main(const Position& pos, Move ttMove, const HistoryTables& history) {
        return MoveGenerator{MovegenStage::TtMove, pos, ttMove, Squares::kNone, &history};
    }

    MoveGenerator MoveGenerator::qsearch(const Position& pos, Square captureSq) {
        const auto initialStage =
            captureSq ? MovegenStage::QsearchGenerateRecaptures : MovegenStage::QsearchGenerateCaptures;
        return MoveGenerator{initialStage, pos, kNullMove, captureSq, nullptr};
    }

    MoveGenerator::MoveGenerator(
        MovegenStage initialStage,
        const Position& pos,
        Move ttMove,
        Square captureSq,
        const HistoryTables* history
    ) :
            m_stage{initialStage}, m_pos{pos}, m_ttMove{ttMove}, m_captureSq{captureSq}, m_history{history} {}

    i32 MoveGenerator::scoreNonCapture(Move move) {
        assert(m_history);
        return m_history->nonCaptureScore(move);
    }

    void MoveGenerator::scoreNonCaptures() {
        assert(m_history);

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
