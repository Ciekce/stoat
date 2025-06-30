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
        void setCuteChessWorkaround(bool enabled);

        void setLimiter(std::unique_ptr<limit::ISearchLimiter> limiter);

        // THIS POINTER WILL BE DANGLING IF setLimiter
        // IS CALLED OR THE SEARCHER IS DESTROYED
        [[nodiscard]] inline limit::ISearchLimiter* limiter() {
            return m_limiter.get();
        }

        void startSearch(
            const Position& pos,
            std::span<const u64> keyHistory,
            util::Instant startTime,
            bool infinite,
            i32 maxDepth,
            std::unique_ptr<limit::ISearchLimiter> limiter
        );
        void stop();

        // THIS REFERENCE WILL BE DANGLING IF setThreads
        // IS CALLED OR THE SEARCHER IS DESTROYED
        [[nodiscard]] ThreadData& mainThread();

        void runBenchSearch(BenchInfo& info, const Position& pos, i32 depth);
        void runDatagenSearch();

        [[nodiscard]] bool isSearching() const;

    private:
        std::vector<ThreadData> m_threads{};

        bool m_silent{};
        bool m_cuteChessWorkaround{};

        mutable std::mutex m_searchMutex{};
        bool m_searching{};

        util::Instant m_startTime{util::Instant::now()};

        util::Barrier m_resetBarrier{2};
        util::Barrier m_idleBarrier{2};

        util::Barrier m_searchEndBarrier{1};

        std::mutex m_stopMutex{};
        std::condition_variable m_stopSignal{};

        std::atomic<u32> m_runningThreads{};

        std::atomic_bool m_stop{};
        std::atomic_bool m_quit{};

        bool m_infinite{};
        std::unique_ptr<limit::ISearchLimiter> m_limiter{};

        movegen::MoveList m_rootMoveList{};

        tt::TTable m_ttable;

        enum class RootStatus {
            kNoLegalMoves = 0,
            kGenerated,
        };

        RootStatus initRootMoves(movegen::MoveList& dst, const Position& pos);

        void runThread(ThreadData& thread);

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

        void report(const ThreadData& bestThread, i32 depth, f64 time);
        void finalReport(f64 time);
    };
} // namespace stoat
