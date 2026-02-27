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

#include <iterator>

namespace stoat {
    ThreadData::ThreadData() {
        keyHistory.reserve(1024);

        stack.resize(kMaxDepth + 1);
        conthist.resize(kMaxDepth + 1);
    }

    void ThreadData::reset(const Position& newRootPos, std::span<const u64> newKeyHistory) {
        rootPos = newRootPos;

        keyHistory.clear();
        keyHistory.reserve(newKeyHistory.size());

        std::ranges::copy(newKeyHistory, std::back_inserter(keyHistory));

        stats.seldepth.store(0);
        stats.nodes.store(0);
    }

    std::pair<Position, ThreadPosGuard<true>> ThreadData::applyMove(i32 ply, const Position& pos, Move move) {
        stack[ply].move = move;
        conthist[ply] = &history.contTable(pos, move);

        keyHistory.push_back(pos.key());

        return std::pair<Position, ThreadPosGuard<true>>{
            std::piecewise_construct,
            std::forward_as_tuple(pos.applyMove(move, nnueState.push())),
            std::forward_as_tuple(keyHistory, nnueState)
        };
    }

    std::pair<Position, ThreadPosGuard<false>> ThreadData::applyNullMove(i32 ply, const Position& pos) {
        stack[ply].move = kNullMove;
        conthist[ply] = nullptr;

        keyHistory.push_back(pos.key());

        return std::pair<Position, ThreadPosGuard<false>>{
            std::piecewise_construct,
            std::forward_as_tuple(pos.applyNullMove()),
            std::forward_as_tuple(keyHistory, nnueState)
        };
    }

    RootMove* ThreadData::findRootMove(Move move) {
        for (u32 idx = pvIdx; idx < rootMoves.size(); ++idx) {
            auto& rootMove = rootMoves[idx];
            assert(rootMove.pv.length > 0);

            if (move == rootMove.pv.moves[0]) {
                return &rootMove;
            }
        }

        return nullptr;
    }
} // namespace stoat
