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

#include <algorithm>
#include <memory>
#include <vector>

#include "util/timer.h"

namespace stoat::limit {
    class ISearchLimiter {
    public:
        virtual ~ISearchLimiter() = default;

        [[nodiscard]] virtual bool stopSoft(usize nodes) = 0;
        [[nodiscard]] virtual bool stopHard(usize nodes) = 0;
    };

    class CompoundLimiter final : public ISearchLimiter {
    public:
        ~CompoundLimiter() final = default;

        template <typename T, typename... Args>
        inline auto addLimiter(Args&&... args) {
            m_limiters.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        }

        [[nodiscard]] inline bool stopSoft(usize nodes) final {
            return std::ranges::any_of(m_limiters, [&](const auto& limiter) { return limiter->stopSoft(nodes); });
        }

        [[nodiscard]] inline bool stopHard(usize nodes) final {
            return std::ranges::any_of(m_limiters, [&](const auto& limiter) { return limiter->stopHard(nodes); });
        }

    private:
        std::vector<std::unique_ptr<ISearchLimiter>> m_limiters{};
    };

    class NodeLimiter final : public ISearchLimiter {
    public:
        explicit NodeLimiter(usize maxNodes);
        ~NodeLimiter() final = default;

        [[nodiscard]] bool stopSoft(usize nodes) final;
        [[nodiscard]] bool stopHard(usize nodes) final;

    private:
        usize m_maxNodes;
    };

    class MoveTimeLimiter final : public ISearchLimiter {
    public:
        MoveTimeLimiter(util::Instant startTime, f64 maxTime);
        ~MoveTimeLimiter() final = default;

        [[nodiscard]] bool stopSoft(usize nodes) final;
        [[nodiscard]] bool stopHard(usize nodes) final;

    private:
        util::Instant m_startTime;
        f64 m_maxTime;
    };

    struct TimeLimits {
        f64 remaining;
        f64 increment;
    };

    class TimeManager final : public ISearchLimiter {
    public:
        TimeManager(util::Instant startTime, const TimeLimits& limits);
        ~TimeManager() final = default;

        [[nodiscard]] bool stopSoft(usize nodes) final;
        [[nodiscard]] bool stopHard(usize nodes) final;

    private:
        util::Instant m_startTime;
        f64 m_maxTime;
    };
} // namespace stoat::limit
