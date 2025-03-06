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

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kCaptures: {
                if (const auto move = selectNext([this](Move move) { return move != m_ttMove; })) {
                    return move;
                }

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kGenerateNonCaptures: {
                if (!m_skipNonCaptures) {
                    movegen::generateNonCaptures(m_moves, m_pos);
                    m_end = m_moves.size();
                }

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::kNonCaptures: {
                if (!m_skipNonCaptures) {
                    if (const auto move = selectNext([this](Move move) { return move != m_ttMove; })) {
                        return move;
                    }
                }

                m_stage = MovegenStage::kEnd;
                return kNullMove;
            }

            case MovegenStage::kQsearchGenerateCaptures: {
                movegen::generateCaptures(m_moves, m_pos);
                m_end = m_moves.size();

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

    MoveGenerator MoveGenerator::main(const Position& pos, Move ttMove) {
        return MoveGenerator{MovegenStage::kTtMove, pos, ttMove};
    }

    MoveGenerator MoveGenerator::qsearch(const Position& pos) {
        const auto initialStage =
            pos.isInCheck() ? MovegenStage::kQsearchEvasionsGenerateCaptures : MovegenStage::kQsearchGenerateCaptures;
        return MoveGenerator{initialStage, pos, kNullMove};
    }

    MoveGenerator::MoveGenerator(MovegenStage initialStage, const Position& pos, Move ttMove) :
            m_stage{initialStage}, m_pos{pos}, m_ttMove{ttMove} {}
} // namespace stoat
