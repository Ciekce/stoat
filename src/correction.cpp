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

#include "correction.h"

namespace stoat {
    void CorrectionHistoryTable::clear() {
        std::memset(&m_blackCastleTable, 0, sizeof(m_blackCastleTable));
        std::memset(&m_whiteCastleTable, 0, sizeof(m_whiteCastleTable));
    }

    void CorrectionHistoryTable::update(const Position& pos, i32 depth, Score searchScore, Score staticEval) {
        const auto bonus = std::clamp((searchScore - staticEval) * depth / 8, -kMaxBonus, kMaxBonus);

        m_blackCastleTable[pos.stm().idx()][pos.blackCastleKey() % kEntries].update(bonus);
        m_whiteCastleTable[pos.stm().idx()][pos.whiteCastleKey() % kEntries].update(bonus);
    }

    i32 CorrectionHistoryTable::correction(const Position& pos) const {
        i32 correction{};

        correction += m_blackCastleTable[pos.stm().idx()][pos.blackCastleKey() % kEntries];
        correction += m_whiteCastleTable[pos.stm().idx()][pos.whiteCastleKey() % kEntries];

        return correction / 16;
    }
} // namespace stoat
