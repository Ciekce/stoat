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
        constexpr f64 kMoveOverhead = 0.01;
    } // namespace

    NodeLimiter::NodeLimiter(usize maxNodes) :
            m_maxNodes{maxNodes} {}

    bool NodeLimiter::stopSoft(usize nodes) {
        return stopHard(nodes);
    }

    bool NodeLimiter::stopHard(usize nodes) {
        return nodes >= m_maxNodes;
    }

    MoveTimeLimiter::MoveTimeLimiter(util::Instant startTime, f64 maxTime) :
            m_startTime{startTime}, m_maxTime{maxTime} {}

    bool MoveTimeLimiter::stopSoft(usize nodes) {
        return m_startTime.elapsed() >= m_maxTime;
    }

    bool MoveTimeLimiter::stopHard(usize nodes) {
        if (nodes == 0 || nodes % kTimeCheckInterval != 0) {
            return false;
        }

        return stopSoft(nodes);
    }

    TimeManager::TimeManager(util::Instant startTime, const TimeLimits& limits) :
            m_startTime{startTime} {
        const auto remaining = limits.remaining - kMoveOverhead;
        const auto maxTime = remaining * 0.05 + limits.increment * 0.5;

        m_maxTime = std::min(maxTime, remaining);
    }

    bool TimeManager::stopSoft(usize nodes) {
        return m_startTime.elapsed() >= m_maxTime;
    }

    bool TimeManager::stopHard(usize nodes) {
        if (nodes == 0 || nodes % kTimeCheckInterval != 0) {
            return false;
        }

        return stopSoft(nodes);
    }
} // namespace stoat::limit
