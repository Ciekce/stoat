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

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "arch.h"
#include "limit.h"
#include "movegen.h"
#include "position.h"
#include "pv.h"
#include "thread.h"
#include "ttable.h"
#include "util/barrier.h"
#include "util/timer.h"

namespace stoat {
    constexpr u32 kDefaultThreadCount = 1;
    constexpr util::Range<u32> kThreadCountRange{1, 2048};

    constexpr u32 kDefaultMultiPv = 1;
    constexpr util::Range<u32> kMultiPvRange{1, 600};

    struct BenchInfo {
        usize nodes{};
        f64 time{};
    };

    class Searcher {
    public:
        explicit Searcher(usize ttSizeMib);
        ~Searcher();

        void newGame();
        void ensureReady();

        void setThreadCount(u32 threadCount);
        void setTtSize(usize mib);
        void setMultiPv(u32 multipv);
        void setMinimal(bool minimal);
        void setCuteChessWorkaround(bool enabled);

        void setLimiter(limit::SearchLimiter limiter);

        void startSearch(
            const Position& pos,
            std::span<const u64> keyHistory,
            util::Instant startTime,
            bool infinite,
            i32 maxDepth
        );
        void stop();

        // Clears all threads, and reallocates main thread data on the current NUMA node.
        // Makes this object unusable for normal searches, just for benching or datagen
        [[nodiscard]] ThreadData& take();

        void runBenchSearch(BenchInfo& info);
        void runDatagenSearch();

        [[nodiscard]] bool isSearching() const;

        [[nodiscard]] u32 threadCount() const;

    private:
        std::vector<std::thread> m_threads{};
        std::vector<std::unique_ptr<ThreadData>> m_threadData{};

        bool m_silent{};
        bool m_minimal{};
        bool m_cuteChessWorkaround{};

        mutable std::mutex m_searchMutex{};
        bool m_searching{};

        util::Instant m_startTime{util::Instant::now()};

        util::Barrier m_initBarrier{2};

        util::Barrier m_resetBarrier{2};
        util::Barrier m_idleBarrier{2};

        util::Barrier m_searchEndBarrier{1};

        std::mutex m_stopMutex{};
        std::condition_variable m_stopSignal{};

        std::atomic<u32> m_runningThreads{};

        std::atomic_bool m_stop{};
        std::atomic_bool m_quit{};

        bool m_infinite{};
        std::optional<limit::SearchLimiter> m_limiter;

        u32 m_targetMultiPv{kDefaultMultiPv};
        u32 m_multiPv{};

        movegen::MoveList m_rootMoveList{};

        tt::TTable m_ttable;

        enum class RootStatus {
            kNoLegalMoves = 0,
            kGenerated,
        };

        RootStatus initRootMoves(movegen::MoveList& dst, const Position& pos);

        void runThread(u32 id);

        [[nodiscard]] inline bool hasStopped() const {
            return m_stop.load(std::memory_order::relaxed);
        }

        void stopThreads();

        void runSearch(ThreadData& thread);

        template <bool kPvNode = false, bool kRootNode = false>
        Score search(
            ThreadData& thread,
            const Position& pos,
            PvList& pv,
            i32 depth,
            i32 ply,
            Score alpha,
            Score beta,
            bool expectedCutnode
        );

        template <>
        Score search<false, true>(
            ThreadData& thread,
            const Position& pos,
            PvList& pv,
            i32 depth,
            i32 ply,
            Score alpha,
            Score beta,
            bool expectedCutnode
        ) = delete;

        template <bool kPvNode = false>
        Score qsearch(ThreadData& thread, const Position& pos, i32 ply, Score alpha, Score beta);

        void reportSingle(const ThreadData& bestThread, u32 pvIdx, i32 depth, f64 time) const;

        void report(const ThreadData& bestThread, i32 depth, f64 time) const;

        const ThreadData& selectThread() const;
        void finalReport(f64 time) const;
    };
} // namespace stoat
