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

#include "thread.h"

namespace stoat {
    ThreadData::ThreadData() {
        keyHistory.reserve(1024);
        stack.resize(kMaxDepth);
    }

    void ThreadData::reset(const Position& newRootPos, std::span<const u64> newKeyHistory) {
        rootPos = newRootPos;

        keyHistory.clear();
        keyHistory.reserve(newKeyHistory.size());

        std::ranges::copy(newKeyHistory, std::back_inserter(keyHistory));

        stats.seldepth.store(0);
        stats.nodes.store(0);

        history.clear();
    }

    std::pair<Position, ThreadPosGuard> ThreadData::applyMove(const Position& pos, Move move) {
        keyHistory.push_back(pos.key());
        return std::pair<Position, ThreadPosGuard>{
            std::piecewise_construct,
            std::forward_as_tuple(pos.applyMove(move)),
            std::forward_as_tuple(keyHistory)
        };
    }
} // namespace stoat
