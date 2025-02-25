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

#include "../../types.h"

#include <utility>
#include <vector>

#include "format.h"

namespace stoat::datagen::format {
    class Stoatpack final : IDataFormat {
    public:
        Stoatpack();
        ~Stoatpack() final = default;

        void startStandard() final;

        void pushUnscored(Move move) final;
        void push(Move move, Score score) final;

        void writeAllWithOutcome(std::ostream& stream, Outcome outcome) final;

    private:
        using ScoredMove = std::pair<u16, i16>;
        static_assert(sizeof(ScoredMove) == sizeof(u16) + sizeof(i16));

        std::vector<u16> m_unscoredMoves{};
        std::vector<ScoredMove> m_moves{};
    };
} // namespace stoat::datagen::format
