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

#include "arch.h"
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
        Score staticEval{};
        i32 depth{};
        Move move{};
        Flag flag{};
        bool pv;
    };

    class TTable {
    public:
        explicit TTable(usize mib);
        ~TTable();

        void resize(usize mib);
        bool finalize(u32 threadCount = 1);

        bool probe(ProbedEntry& dst, u64 key, i32 ply) const;
        void put(u64 key, Score score, Score staticEval, Move move, i32 depth, i32 ply, Flag flag, bool pv);

        inline void age() {
            m_age = (m_age + 1) % Entry::kAgeCycle;
        }

        void clear(u32 threadCount = 1);

        [[nodiscard]] u32 fullPermille() const;

        inline void prefetch(u64 key) {
            __builtin_prefetch(&m_entries[index(key)]);
        }

    private:
        struct __attribute__((packed)) Entry {
            static constexpr u32 kAgeBits = 5;
            static constexpr u32 kAgeCycle = 1 << kAgeBits;

            u16 key;
            i16 score;
            i16 staticEval;
            u16 move;
            u8 depth;
            u8 agePvFlag;

            [[nodiscard]] inline u32 age() const {
                return static_cast<u32>(agePvFlag >> 3);
            }

            [[nodiscard]] inline bool pv() const {
                return agePvFlag & 0b100;
            }

            [[nodiscard]] inline Flag flag() const {
                return static_cast<Flag>(agePvFlag & 0x3);
            }

            inline void setAgePvFlag(u32 age, bool pv, Flag flag) {
                assert(age < (1 << kAgeBits));
                agePvFlag = (age << 3) | (static_cast<u32>(pv) << 2) | static_cast<u32>(flag);
            }
        };

        static_assert(sizeof(Entry) == 10);

        static constexpr usize kSmallPageSize = 4096;
        static constexpr auto kDefaultStorageAlignment = std::max(kCacheLineSize, kSmallPageSize);

        bool m_pendingInit{};

        // is this an owning raw pointer? :fearful:
        // yes :pensive:
        Entry* m_entries{};
        usize m_entryCount{};

        u32 m_age{};

        [[nodiscard]] constexpr usize index(u64 key) const {
            return static_cast<usize>((static_cast<u128>(key) * static_cast<u128>(m_entryCount)) >> 64);
        }
    };
} // namespace stoat::tt
