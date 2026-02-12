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

    TimeManager::TimeManager(const TimeLimits& limits, u32 moveOverheadMs, u32 moveCount) {
        const auto moveOverhead = static_cast<f64>(moveOverheadMs) / 1000.0;

        const auto remaining = std::max(limits.remaining - moveOverhead, 0.0);
        const auto extra = std::max(limits.byoyomi - moveOverhead, 0.0);

        const auto baseTime =
            std::min(remaining * (moveCount + 1) / 1000 + limits.increment * (moveCount + 1) / 100, remaining) + extra;
        const auto optTime = baseTime * 0.6;

        m_maxTime = remaining * 0.6 + extra;
        m_optTime = std::min(optTime, m_maxTime);
    }

    void TimeManager::update(i32 depth, usize totalNodes, const RootMove& pvMove) {
        m_scale = 1.0;

        if (depth <= 5) {
            return;
        }

        const auto bestMoveNodeFraction = static_cast<f64>(pvMove.nodes) / static_cast<f64>(totalNodes);
        m_scale *= 2.2 - bestMoveNodeFraction * 1.6;
    }

    bool TimeManager::stopSoft(f64 time) const {
        return time >= m_optTime * m_scale;
    }

    bool TimeManager::stopHard(f64 time) const {
        return time >= m_maxTime;
    }

    SearchLimiter::SearchLimiter(util::Instant startTime) :
            m_startTime{startTime} {}

    bool SearchLimiter::setHardNodes(usize nodes) {
        if (m_hardNodes) {
            return false;
        }

        m_hardNodes = nodes;
        return true;
    }

    bool SearchLimiter::setSoftNodes(usize nodes) {
        if (m_softNodes) {
            return false;
        }

        m_softNodes = nodes;
        return true;
    }

    bool SearchLimiter::setMoveTime(f64 time) {
        if (m_moveTime) {
            return false;
        }

        m_moveTime = time;
        return true;
    }

    bool SearchLimiter::setTournamentTime(const TimeLimits& limits, u32 moveOverheadMs, u32 moveCount) {
        if (m_timeManager) {
            return false;
        }

        m_timeManager = TimeManager{limits, moveOverheadMs, moveCount};
        return true;
    }

    void SearchLimiter::update(i32 depth, usize totalNodes, const RootMove& pvMove) {
        if (m_timeManager) {
            m_timeManager->update(depth, totalNodes, pvMove);
        }
    }

    bool SearchLimiter::stopSoft(usize nodes) const {
        if (m_softNodes && nodes >= *m_softNodes) {
            return true;
        }

        if (m_moveTime || m_timeManager) {
            const auto time = m_startTime.elapsed();

            if (m_moveTime && time >= *m_moveTime) {
                return true;
            }

            if (m_timeManager && m_timeManager->stopSoft(time)) {
                return true;
            }
        }

        return false;
    }

    bool SearchLimiter::stopHard(usize nodes) const {
        if (m_hardNodes && nodes >= *m_hardNodes) {
            return true;
        }

        if (nodes > 0 && nodes % kTimeCheckInterval == 0 && (m_moveTime || m_timeManager)) {
            const auto time = m_startTime.elapsed();

            if (m_moveTime && time >= *m_moveTime) {
                return true;
            }

            if (m_timeManager && m_timeManager->stopHard(time)) {
                return true;
            }
        }

        return false;
    }
} // namespace stoat::limit
