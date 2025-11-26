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

#include "move.h"
#include "util/multi_array.h"
#include "util/timer.h"

namespace stoat::limit {
    class ISearchLimiter {
    public:
        virtual ~ISearchLimiter() = default;

        virtual void addMoveNodes([[maybe_unused]] Move move, [[maybe_unused]] usize nodes) {}

        virtual void update([[maybe_unused]] i32 depth, [[maybe_unused]] Move bestMove) {}

        [[nodiscard]] virtual bool stopSoft(usize nodes) = 0;
        [[nodiscard]] virtual bool stopHard(usize nodes) = 0;
    };

    class CompoundLimiter final : public ISearchLimiter {
    public:
        ~CompoundLimiter() final = default;

        template <typename T, typename... Args>
        inline void addLimiter(Args&&... args) {
            m_limiters.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        }

        inline void addMoveNodes(Move move, usize nodes) final {
            for (auto& limiter : m_limiters) {
                limiter->addMoveNodes(move, nodes);
            }
        }

        inline void update(i32 depth, Move bestMove) final {
            for (auto& limiter : m_limiters) {
                limiter->update(depth, bestMove);
            }
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

    class SoftNodeLimiter final : public ISearchLimiter {
    public:
        explicit SoftNodeLimiter(usize optNodes, usize maxNodes);
        ~SoftNodeLimiter() final = default;

        [[nodiscard]] bool stopSoft(usize nodes) final;
        [[nodiscard]] bool stopHard(usize nodes) final;

    private:
        usize m_optNodes;
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
        f64 byoyomi;
    };

    class TimeManager final : public ISearchLimiter {
    public:
        TimeManager(util::Instant startTime, const TimeLimits& limits, u32 moveOverheadMs);
        ~TimeManager() final = default;

        void addMoveNodes(Move move, usize nodes) final;

        void update(i32 depth, Move bestMove) final;

        [[nodiscard]] bool stopSoft(usize nodes) final;
        [[nodiscard]] bool stopHard(usize nodes) final;

    private:
        util::Instant m_startTime;

        f64 m_optTime;
        f64 m_maxTime;

        f64 m_scale{1.0};

        // [promo][from][to]
        util::MultiArray<usize, 2, Squares::kCount, Squares::kCount> m_nonDrop{};
        // [dropped piece type][drop square]
        util::MultiArray<usize, PieceTypes::kCount, Squares::kCount> m_drop{};

        usize m_totalNodes{};
    };
} // namespace stoat::limit
