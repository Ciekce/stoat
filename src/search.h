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
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "arch.h"
#include "movegen.h"
#include "position.h"
#include "pv.h"
#include "util/barrier.h"
#include "util/timer.h"

namespace stoat {
    struct SearchStats {
        SearchStats() = default;

        inline SearchStats(const SearchStats& other) {
            *this = other;
        }

        std::atomic<i32> seldepth{};
        std::atomic<usize> nodes{};

        SearchStats& operator=(const SearchStats& other) {
            seldepth.store(other.seldepth);
            nodes.store(other.nodes);

            return *this;
        }
    };

    class ThreadPosGuard {
    public:
        explicit ThreadPosGuard(std::vector<u64>& keyHistory) :
                m_keyHistory{keyHistory} {}

        ThreadPosGuard(const ThreadPosGuard&) = delete;
        ThreadPosGuard(ThreadPosGuard&&) = delete;

        inline ~ThreadPosGuard() {
            m_keyHistory.pop_back();
        }

    private:
        std::vector<u64>& m_keyHistory;
    };

    struct StackFrame {
        PvList pv{};
    };

    struct alignas(CacheLineSize) ThreadData {
        ThreadData();

        u32 id{};
        std::thread thread{};

        i32 maxDepth{};

        Position rootPos{};
        std::vector<u64> keyHistory{};

        SearchStats stats{};

        i32 rootDepth{};
        i32 depthCompleted{};

        Score lastScore{};
        PvList lastPv{};

        std::vector<StackFrame> stack{};

        [[nodiscard]] inline u32 isMainThread() const {
            return id == 0;
        }

        [[nodiscard]] inline i32 loadSeldepth() const {
            return stats.seldepth.load(std::memory_order::relaxed);
        }

        inline void updateSeldepth(i32 v) {
            if (v > loadSeldepth()) {
                stats.seldepth.store(v, std::memory_order::relaxed);
            }
        }

        inline void resetSeldepth() {
            stats.seldepth.store(0);
        }

        [[nodiscard]] inline usize loadNodes() const {
            return stats.nodes.load(std::memory_order::relaxed);
        }

        inline void incNodes() {
            stats.nodes.fetch_add(1, std::memory_order::relaxed);
        }

        void reset(const Position& newRootPos, std::span<const u64> newKeyHistory);

        [[nodiscard]] std::pair<Position, ThreadPosGuard> applyMove(const Position& pos, Move move);
    };

    class Searcher {
    public:
        Searcher();
        ~Searcher();

        void newGame();
        void ensureReady();

        void setThreads(u32 threadCount);

        void startSearch(const Position& pos, std::span<const u64> keyHistory, util::Instant startTime, i32 maxDepth);
        void stop();

        [[nodiscard]] bool isSearching() const;

    private:
        std::vector<ThreadData> m_threads{};

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

        movegen::MoveList m_rootMoves{};

        enum class RootStatus {
            kNoLegalMoves = 0,
            kGenerated,
        };

        RootStatus initRootMoves(const Position& pos);

        void runThread(ThreadData& thread);

        [[nodiscard]] inline bool hasStopped() const {
            return m_stop.load(std::memory_order::relaxed);
        }

        void stopThreads();

        void runSearch(ThreadData& thread);

        [[nodiscard]] inline bool isLegalRootMove(Move move) const {
            return std::ranges::find(m_rootMoves, move) != m_rootMoves.end();
        }

        template <bool kRootNode = false>
        Score search(ThreadData& thread, const Position& pos, PvList& pv, i32 depth, i32 ply);

        void report(const ThreadData& bestThread, f64 time);
        void finalReport(f64 time);
    };
} // namespace stoat
