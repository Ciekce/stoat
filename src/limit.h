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

#include <optional>

#include "root_move.h"
#include "util/timer.h"

namespace stoat::limit {
    struct TimeLimits {
        f64 remaining;
        f64 increment;
        f64 byoyomi;
    };

    class TimeManager {
    public:
        TimeManager(const TimeLimits& limits, u32 moveOverheadMs, u32 moveCount);

        void update(i32 depth, usize totalNodes, const RootMove& pvMove);

        [[nodiscard]] bool stopSoft(f64 time) const;
        [[nodiscard]] bool stopHard(f64 time) const;

    private:
        f64 m_optTime;
        f64 m_maxTime;

        f64 m_scale{1.0};
    };

    class SearchLimiter {
    public:
        explicit SearchLimiter(util::Instant startTime);

        bool setHardNodes(usize nodes);
        bool setSoftNodes(usize nodes);

        bool setMoveTime(f64 time);

        bool setTournamentTime(const TimeLimits& limits, u32 moveOverheadMs, u32 moveCount);

        void update(i32 depth, usize totalNodes, const RootMove& pvMove);

        [[nodiscard]] bool stopSoft(usize nodes) const;
        [[nodiscard]] bool stopHard(usize nodes) const;

    private:
        util::Instant m_startTime;

        std::optional<usize> m_hardNodes;
        std::optional<usize> m_softNodes;

        std::optional<f64> m_moveTime;

        std::optional<TimeManager> m_timeManager;
    };
} // namespace stoat::limit
