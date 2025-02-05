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

#include "position.h"
#include "util/multi_array.h"

namespace stoat {
    class CorrectionHistoryTable {
    public:
        void clear();

        void update(const Position& pos, i32 depth, Score searchScore, Score staticEval);

        [[nodiscard]] Score correct(const Position& pos, Score score) const;

    private:
        static constexpr usize kHashEntries = 16384;

        static constexpr i32 kGrain = 256;
        static constexpr i32 kWeightScale = 256;
        static constexpr i32 kMax = kGrain * 32;

        struct Entry {
            i16 value{};

            void update(i32 scaledError, i32 newWeight);

            [[nodiscard]] inline operator i32() const {
                return value / kGrain;
            }
        };

        util::MultiArray<Entry, 2, kHashEntries> m_pawnTable{};
    };
} // namespace stoat
