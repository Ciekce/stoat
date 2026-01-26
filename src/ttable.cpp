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
#include <thread>
#include <vector>

#ifndef _WIN32
    #include <sys/mman.h>
#endif

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

    bool TTable::finalize(u32 threadCount) {
        if (!m_pendingInit) {
            return false;
        }

        m_pendingInit = false;

        if (!m_entries) {
#ifdef MADV_HUGEPAGE
            //TODO handle 1GiB huge pages?
            static constexpr usize kHugePageSize = 2 * 1024 * 1024;

            const auto size = m_entryCount * sizeof(Entry);
            const auto alignment = size >= kHugePageSize ? kHugePageSize : kDefaultStorageAlignment;
#else
            const auto alignment = kDefaultStorageAlignment;
#endif

            m_entries = util::alignedAlloc<Entry>(alignment, m_entryCount);

            if (!m_entries) {
                fmt::println(stderr, "Failed to reallocate TT - out of memory?");
                std::terminate();
            }

#ifdef MADV_HUGEPAGE
            madvise(m_entries, size, MADV_HUGEPAGE);
#endif
        }

        clear(threadCount);

        return true;
    }

    bool TTable::probe(ProbedEntry& dst, u64 key, i32 ply) const {
        assert(!m_pendingInit);

        const auto entry = m_entries[index(key)];

        if (entry.key == packEntryKey(key)) {
            dst.score = scoreFromTt(static_cast<Score>(entry.score), ply);
            dst.staticEval = static_cast<Score>(entry.staticEval);
            dst.move = Move::fromRaw(entry.move);
            dst.depth = static_cast<i32>(entry.depth);
            dst.flag = entry.flag();
            dst.pv = entry.pv();

            return true;
        }

        return false;
    }

    void TTable::put(u64 key, Score score, [[maybe_unused]] Score staticEval, Move move, i32 depth, i32 ply, Flag flag, bool pv) {
        assert(!m_pendingInit);

        assert(depth >= 0);
        assert(depth <= kMaxDepth);

        const auto packedKey = packEntryKey(key);

        auto& slot = m_entries[index(key)];
        auto entry = slot;

        const bool replace =
            flag == Flag::kExact || packedKey != entry.key || entry.age() != m_age || depth + 4 > entry.depth;

        if (!replace) {
            return;
        }

        if (move || entry.key != packedKey) {
            entry.move = move.raw();
        }

        entry.key = packedKey;
        entry.score = static_cast<i16>(scoreToTt(score, ply));
        entry.staticEval = static_cast<i16>(staticEval);
        entry.depth = static_cast<u8>(depth);
        entry.setAgePvFlag(m_age, pv, flag);

        slot = entry;
    }

    void TTable::clear(u32 threadCount) {
        assert(!m_pendingInit);

        std::vector<std::thread> threads{};
        threads.reserve(threadCount);

        fmt::println("info string Clearing the TT with {} threads", threadCount);

        const auto chunkSize = (m_entryCount + threadCount - 1) / threadCount;

        for (u32 i = 0; i < threadCount; ++i) {
            threads.emplace_back([this, chunkSize, i] {
                const auto start = chunkSize * i;
                const auto end = std::min(start + chunkSize, m_entryCount);

                const auto count = end - start;

                std::memset(&m_entries[start], 0, count * sizeof(Entry));
            });
        }

        m_age = 0;

        for (auto& thread : threads) {
            thread.join();
        }
    }

    u32 TTable::fullPermille() const {
        assert(!m_pendingInit);

        u32 filledEntries{};

        for (usize i = 0; i < 1000; ++i) {
            const auto entry = m_entries[i];
            if (entry.flag() != Flag::kNone && entry.age() == m_age) {
                ++filledEntries;
            }
        }

        return filledEntries;
    }
} // namespace stoat::tt
