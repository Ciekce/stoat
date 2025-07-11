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
#include <cstring>

#include "core.h"
#include "position.h"
#include "util/multi_array.h"

namespace stoat {
    class CorrectionHistoryTable {
    public:
        inline void clear() {
            std::memset(&m_castleTable, 0, sizeof(m_castleTable));
        }

        inline void update(const Position& pos, i32 depth, Score searchScore, Score staticEval) {
            const auto bonus = std::clamp((searchScore - staticEval) * depth / 8, -kMaxBonus, kMaxBonus);
            m_castleTable[pos.stm().idx()][pos.castleKey() % kEntries].update(bonus);
        }

        [[nodiscard]] inline Score correct(const Position& pos, Score score) const {
            i32 correction{};

            correction += m_castleTable[pos.stm().idx()][pos.castleKey() % kEntries];

            score += correction / 16;

            return std::clamp(score, -kScoreWin + 1, kScoreWin - 1);
        }

    private:
        static constexpr usize kEntries = 16384;

        static constexpr i32 kLimit = 1024;
        static constexpr i32 kMaxBonus = kLimit / 4;

        struct Entry {
            i16 value{};

            inline void update(i32 bonus) {
                value += bonus - value * std::abs(bonus) / kLimit;
            }

            [[nodiscard]] inline operator i32() const {
                return value;
            }
        };

        util::MultiArray<Entry, 2, kEntries> m_castleTable{};
    };
} // namespace stoat
