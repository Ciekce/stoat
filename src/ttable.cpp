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

#include "ttable.h"

#include <cstring>

#include "arch.h"
#include "core.h"
#include "util/align.h"

namespace stoat::tt {
    namespace {
        [[nodiscard]] constexpr Score scoreToTt(Score score, i32 ply) {
            if (score < -kScoreWin) {
                return score - ply;
            } else if (score > kScoreWin) {
                return score + ply;
            } else {
                return score;
            }
        }

        [[nodiscard]] constexpr Score scoreFromTt(Score score, i32 ply) {
            if (score < -kScoreWin) {
                return score + ply;
            } else if (score > kScoreWin) {
                return score - ply;
            } else {
                return score;
            }
        }

        [[nodiscard]] constexpr u16 packEntryKey(u64 key) {
            return static_cast<u16>(key);
        }
    } // namespace

    TTable::TTable(usize mib) {
        resize(mib);
    }

    TTable::~TTable() {
        if (m_entries) {
            util::alignedFree(m_entries);
        }
    }

    void TTable::resize(usize mib) {
        const auto bytes = mib * 1024 * 1024;
        const auto entries = bytes / sizeof(Entry);

        if (m_entryCount != entries) {
            if (m_entries) {
                util::alignedFree(m_entries);
            }

            m_entries = nullptr;
            m_entryCount = entries;
        }

        m_pendingInit = true;
    }

    bool TTable::finalize() {
        if (!m_pendingInit) {
            return false;
        }

        m_pendingInit = false;
        m_entries = util::alignedAlloc<Entry>(kCacheLineSize, m_entryCount);

        if (!m_entries) {
            std::cerr << "Failed to reallocate TT - out of memory?" << std::endl;
            std::terminate();
        }

        clear();

        return true;
    }

    bool TTable::probe(ProbedEntry& dst, u64 key, i32 ply) const {
        assert(!m_pendingInit);

        const auto entry = m_entries[index(key)];

        if (entry.key == packEntryKey(key)) {
            dst.score = scoreFromTt(static_cast<Score>(entry.score), ply);
            dst.depth = static_cast<i32>(entry.depth);
            dst.flag = entry.flag;

            return true;
        }

        return false;
    }

    void TTable::put(u64 key, Score score, i32 depth, i32 ply, Flag flag) {
        assert(!m_pendingInit);

        assert(depth >= 0);
        assert(depth <= kMaxDepth);

        auto& entry = m_entries[index(key)];

        auto newEntry = entry;

        newEntry.key = packEntryKey(key);
        newEntry.score = static_cast<i16>(scoreToTt(score, ply));
        newEntry.depth = static_cast<u8>(depth);
        newEntry.flag = flag;

        entry = newEntry;
    }

    void TTable::clear() {
        assert(!m_pendingInit);
        std::memset(m_entries, 0, m_entryCount * sizeof(Entry));
    }

    u32 TTable::fullPermille() const {
        assert(!m_pendingInit);

        u32 filledEntries{};

        for (usize i = 0; i < 1000; ++i) {
            const auto entry = m_entries[i];
            if (entry.flag != Flag::kNone) {
                ++filledEntries;
            }
        }

        return filledEntries;
    }
} // namespace stoat::tt
