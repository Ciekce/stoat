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

#include "limit.h"

namespace stoat::limit {
    namespace {
        constexpr usize kTimeCheckInterval = 2048;
    } // namespace

    NodeLimiter::NodeLimiter(usize maxNodes) :
            m_maxNodes{maxNodes} {}

    bool NodeLimiter::stopSoft(usize nodes) {
        return stopHard(nodes);
    }

    bool NodeLimiter::stopHard(usize nodes) {
        return nodes >= m_maxNodes;
    }

    SoftNodeLimiter::SoftNodeLimiter(usize optNodes, usize maxNodes) :
            m_optNodes{optNodes}, m_maxNodes{maxNodes} {
        if (m_optNodes > m_maxNodes) {
            m_optNodes = m_maxNodes;
        }
    }

    bool SoftNodeLimiter::stopSoft(usize nodes) {
        return nodes >= m_optNodes;
    }

    bool SoftNodeLimiter::stopHard(usize nodes) {
        return nodes >= m_maxNodes;
    }

    MoveTimeLimiter::MoveTimeLimiter(util::Instant startTime, f64 maxTime) :
            m_startTime{startTime}, m_maxTime{maxTime} {}

    bool MoveTimeLimiter::stopSoft([[maybe_unused]] usize nodes) {
        return m_startTime.elapsed() >= m_maxTime;
    }

    bool MoveTimeLimiter::stopHard(usize nodes) {
        if (nodes == 0 || nodes % kTimeCheckInterval != 0) {
            return false;
        }

        return stopSoft(nodes);
    }

    TimeManager::TimeManager(util::Instant startTime, const TimeLimits& limits, u32 moveOverheadMs) :
            m_startTime{startTime} {
        fmt::println("info string move overhead: {} ms", moveOverheadMs);

        const auto moveOverhead = static_cast<f64>(moveOverheadMs) / 1000.0;

        const auto remaining = std::max(limits.remaining - moveOverhead, 0.0);
        const auto extra = std::max(limits.byoyomi - moveOverhead, 0.0);

        const auto baseTime = std::min(remaining * 0.05 + limits.increment * 0.5, remaining) + extra;
        const auto optTime = baseTime * 0.6;

        m_maxTime = remaining * 0.6 + extra;
        m_optTime = std::min(optTime, m_maxTime);
    }

    void TimeManager::addMoveNodes(Move move, usize nodes) {
        if (move.isDrop()) {
            m_drop[move.dropPiece().idx()][move.to().idx()] += nodes;
        } else {
            m_nonDrop[move.isPromo()][move.from().idx()][move.to().idx()] += nodes;
        }

        m_totalNodes += nodes;
    }

    void TimeManager::update(i32 depth, Move bestMove) {
        m_scale = 1.0;

        if (depth <= 5) {
            return;
        }

        const auto bestMoveNodes = [&] {
            if (bestMove.isDrop()) {
                return m_drop[bestMove.dropPiece().idx()][bestMove.to().idx()];
            } else {
                return m_nonDrop[bestMove.isPromo()][bestMove.from().idx()][bestMove.to().idx()];
            }
        }();

        const auto bestMoveNodeFraction = static_cast<f64>(bestMoveNodes) / static_cast<f64>(m_totalNodes);
        m_scale *= 2.2 - bestMoveNodeFraction * 1.6;
    }

    bool TimeManager::stopSoft([[maybe_unused]] usize nodes) {
        return util::Instant::now() >= m_startTime + m_optTime * m_scale;
    }

    bool TimeManager::stopHard(usize nodes) {
        if (nodes == 0 || nodes % kTimeCheckInterval != 0) {
            return false;
        }

        return m_startTime.elapsed() >= m_maxTime;
    }
} // namespace stoat::limit
