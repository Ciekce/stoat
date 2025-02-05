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

#include <algorithm>
#include <cstring>

namespace stoat {
    namespace {
        template <std::integral auto kOne>
        constexpr decltype(kOne) ilerp(decltype(kOne) a, decltype(kOne) b, decltype(kOne) t) {
            return (a * (kOne - t) + b * t) / kOne;
        }
    } // namespace

    void CorrectionHistoryTable::clear() {
        std::memset(&m_pawnTable, 0, sizeof(m_pawnTable));
    }

    void CorrectionHistoryTable::update(const Position& pos, i32 depth, Score searchScore, Score staticEval) {
        const auto scaledError = static_cast<i32>((searchScore - staticEval) * kGrain);
        const auto newWeight = static_cast<i32>(std::min(depth + 1, 16));

        m_pawnTable[pos.stm().idx()][pos.pawnKey() % kHashEntries].update(scaledError, newWeight);
    }

    Score CorrectionHistoryTable::correct(const Position& pos, Score score) const {
        i32 correction{};

        correction += m_pawnTable[pos.stm().idx()][pos.pawnKey() % kHashEntries];

        return std::clamp(score + correction, -kScoreWin + 1, kScoreWin - 1);
    }

    void CorrectionHistoryTable::Entry::update(i32 scaledError, i32 newWeight) {
        const auto v = ilerp<kWeightScale>(value, scaledError, newWeight);
        value = static_cast<i16>(std::clamp(v, -kMax, kMax));
    }
} // namespace stoat
