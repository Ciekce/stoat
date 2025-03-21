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

#include "core.h"
#include "move.h"
#include "util/range.h"

namespace stoat::tt {
    constexpr usize kDefaultTtSizeMib = 64;
    constexpr util::Range<usize> kTtSizeRange{1, 131072};

    enum class Flag : u8 {
        kNone = 0,
        kUpperBound,
        kLowerBound,
        kExact,
    };

    struct ProbedEntry {
        Score score{};
        i32 depth{};
        Move move{};
        Flag flag{};
    };

    class TTable {
    public:
        explicit TTable(usize mib);
        ~TTable();

        void resize(usize mib);
        bool finalize();

        bool probe(ProbedEntry& dst, u64 key, i32 ply) const;
        void put(u64 key, Score score, Move move, i32 depth, i32 ply, Flag flag);

        void clear();

        [[nodiscard]] u32 fullPermille() const;

        inline void prefetch(u64 key) {
            __builtin_prefetch(&m_entries[index(key)]);
        }

    private:
        struct alignas(8) Entry {
            u16 key;
            i16 score;
            Move move;
            u8 depth;
            Flag flag;
        };

        static_assert(sizeof(Entry) == 8);

        bool m_pendingInit{};

        // is this an owning raw pointer? :fearful:
        // yes :pensive:
        Entry* m_entries{};
        usize m_entryCount{};

        [[nodiscard]] constexpr usize index(u64 key) const {
            return static_cast<usize>((static_cast<u128>(key) * static_cast<u128>(m_entryCount)) >> 64);
        }
    };
} // namespace stoat::tt
