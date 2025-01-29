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

            case MovegenStage::Generate: {
                movegen::generateAll(m_moves, m_pos);
                m_end = m_moves.size();

                ++m_stage;
                [[fallthrough]];
            }

            case MovegenStage::All: {
                if (const auto move = selectNext([this](Move move) { return move != m_ttMove; })) {
                    return move;
                }

                m_stage = MovegenStage::End;
                return kNullMove;
            }

            default:
                return kNullMove;
        }
    }

    MoveGenerator MoveGenerator::create(const Position& pos, Move ttMove) {
        return MoveGenerator{MovegenStage::TtMove, pos, ttMove};
    }

    MoveGenerator::MoveGenerator(MovegenStage initialStage, const Position& pos, Move ttMove) :
            m_stage{initialStage}, m_pos{pos}, m_ttMove{ttMove} {}
} // namespace stoat
