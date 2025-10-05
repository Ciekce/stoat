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

#include <cmath>

namespace stoat {
    void CorrectionHistory::clear() {
        std::memset(&m_tables, 0, sizeof(m_tables));
    }

    void CorrectionHistory::update(
        const Position& pos,
        i32 depth,
        Score searchScore,
        Score staticEval,
        f64 complexityFactor
    ) {
        auto& tables = m_tables[pos.stm().idx()];

        const auto bonus = std::clamp(
            static_cast<i32>((searchScore - staticEval) * depth / 8 * complexityFactor),
            -kMaxBonus,
            kMaxBonus
        );

        tables.castle[pos.castleKey() % kEntries].update(bonus);
        tables.cavalry[pos.cavalryKey() % kEntries].update(bonus);
        tables.hand[pos.kingHandKey() % kEntries].update(bonus);
        tables.kpr[pos.kprKey() % kEntries].update(bonus);
    }

    i32 CorrectionHistory::correction(const Position& pos) const {
        const auto& tables = m_tables[pos.stm().idx()];

        i32 correction{};

        correction += 128 * tables.castle[pos.castleKey() % kEntries];
        correction += 128 * tables.cavalry[pos.cavalryKey() % kEntries];
        correction += 128 * tables.hand[pos.kingHandKey() % kEntries];
        correction += 128 * tables.kpr[pos.kprKey() % kEntries];

        return correction / 2048;
    }
} // namespace stoat
