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
        std::span<PlayedMove> moves,
        i32 ply,
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

        const auto updateCont = [&](i32 i) {
            if (ply <= i) {
                return;
            }

            const auto [move1, piece1] = moves[ply - 1];
            const auto [move2, piece2] = moves[ply - 1 - i];

            if (move1 == kNullMove || move2 == kNullMove) {
                return;
            }

            tables
                .cont[move2.isDrop()][piece2.idx()][move2.to().idx()] //
                     [move1.isDrop()][piece1.idx()][move1.to().idx()]
                .update(bonus);
        };

        tables.castle[pos.castleKey() % kEntries].update(bonus);
        tables.cavalry[pos.cavalryKey() % kEntries].update(bonus);
        tables.hand[pos.kingHandKey() % kEntries].update(bonus);
        tables.kpr[pos.kprKey() % kEntries].update(bonus);

        updateCont(1);
    }

    i32 CorrectionHistory::correction(const Position& pos, std::span<PlayedMove> moves, i32 ply) const {
        const auto& tables = m_tables[pos.stm().idx()];

        const auto contCorrection = [&](i32 i) -> i32 {
            if (ply <= i) {
                return 0;
            }

            const auto [move1, piece1] = moves[ply - 1];
            const auto [move2, piece2] = moves[ply - 1 - i];

            if (move1 == kNullMove || move2 == kNullMove) {
                return 0;
            }

            return tables.cont[move2.isDrop()][piece2.idx()][move2.to().idx()] //
                              [move1.isDrop()][piece1.idx()][move1.to().idx()];
        };

        i32 correction{};

        correction += 128 * tables.castle[pos.castleKey() % kEntries];
        correction += 128 * tables.cavalry[pos.cavalryKey() % kEntries];
        correction += 128 * tables.hand[pos.kingHandKey() % kEntries];
        correction += 128 * tables.kpr[pos.kprKey() % kEntries];

        correction += 128 * contCorrection(1);

        return correction / 2048;
    }
} // namespace stoat
