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

#include "stoatpack.h"

#include <array>

namespace stoat::datagen::format {
    Stoatpack::Stoatpack() {
        m_unscoredMoves.reserve(16);
        m_moves.reserve(256);
    }

    void Stoatpack::startStandard() {
        m_unscoredMoves.clear();
        m_moves.clear();
    }

    void Stoatpack::pushUnscored(Move move) {
        assert(m_moves.empty());
        m_unscoredMoves.push_back(move.raw());
    }

    void Stoatpack::push(Move move, Score score) {
        assert(std::abs(score) <= kScoreInf);
        m_moves.emplace_back(move.raw(), static_cast<i16>(score));
    }

    void Stoatpack::writeAllWithOutcome(std::ostream& stream, Outcome outcome) {
        static constexpr ScoredMove kNullTerminator = {0, 0};

        static constexpr u8 kStandardType = 0;

        const u8 wdlType = kStandardType | (static_cast<u8>(outcome) << 6);
        stream.write(reinterpret_cast<const char*>(&wdlType), sizeof(wdlType));

        const u16 unscoredCount = m_unscoredMoves.size();
        stream.write(reinterpret_cast<const char*>(&unscoredCount), sizeof(unscoredCount));
        stream.write(reinterpret_cast<const char*>(m_unscoredMoves.data()), m_unscoredMoves.size() * sizeof(u16));

        stream.write(reinterpret_cast<const char*>(m_moves.data()), m_moves.size() * sizeof(ScoredMove));
        stream.write(reinterpret_cast<const char*>(&kNullTerminator), sizeof(kNullTerminator));
    }
} // namespace stoat::datagen::format
