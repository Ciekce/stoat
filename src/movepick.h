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

#include <compare>

#include "move.h"
#include "movegen.h"
#include "position.h"

namespace stoat {
    enum class MovegenStage : i32 {
        TtMove,
        Generate,
        All,
        End,
    };

    constexpr MovegenStage& operator++(MovegenStage& v) {
        v = static_cast<MovegenStage>(static_cast<i32>(v) + 1);
        return v;
    }

    [[nodiscard]] constexpr std::strong_ordering operator<=>(MovegenStage a, MovegenStage b) {
        return static_cast<i32>(a) <=> static_cast<i32>(b);
    }

    class MoveGenerator {
    public:
        [[nodiscard]] Move next();

        [[nodiscard]] inline MovegenStage stage() {
            return m_stage;
        }

        [[nodiscard]] static MoveGenerator create(const Position& pos, Move ttMove);

    private:
        MoveGenerator(MovegenStage initialStage, const Position& pos, Move ttMove);

        [[nodiscard]] inline Move selectNext(auto predicate) {
            while (m_idx < m_end) {
                const auto move = m_moves[m_idx++];
                if (predicate(move)) {
                    return move;
                }
            }

            return kNullMove;
        }

        MovegenStage m_stage;

        const Position& m_pos;
        movegen::MoveList m_moves{};

        Move m_ttMove;

        usize m_idx{};
        usize m_end{};
    };
} // namespace stoat
