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
#include <thread>
#include <vector>

#include "core.h"
#include "position.h"
#include "pv.h"

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
        Move move{};
    };

    struct alignas(kCacheLineSize) ThreadData {
        ThreadData();

        u32 id{};
        std::thread thread{};

        i32 maxDepth{};

        bool datagen{false};

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

        [[nodiscard]] std::pair<Position, ThreadPosGuard> applyMove(i32 ply, const Position& pos, Move move);
        [[nodiscard]] std::pair<Position, ThreadPosGuard> applyNullMove(i32 ply, const Position& pos);
    };
} // namespace stoat
