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

#include "../types.h"

#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "../core.h"
#include "../position.h"
#include "../pv.h"
#include "../search.h"

namespace stoat::protocol {
    struct EngineState {
        Position pos{Position::startpos()};
        std::vector<u64> keyHistory{};

        Searcher* searcher{};
    };

    struct CpDisplayScore {
        Score score;
    };

    struct MateDisplayScore {
        i32 plies;
    };

    using DisplayScore = std::variant<CpDisplayScore, MateDisplayScore>;

    struct SearchInfo {
        i32 depth;
        std::optional<i32> seldepth{};
        std::optional<f64> timeSec{};
        usize nodes;
        DisplayScore score;
        const PvList& pv;
        std::optional<u32> hashfull{};
    };

    enum class CommandResult {
        kContinue = 0,
        kQuit,
        kUnknown,
    };

    class IProtocolHandler {
    public:
        virtual ~IProtocolHandler() = default;

        virtual void printInitialInfo() const = 0;

        // gui -> engine
        [[nodiscard]] virtual CommandResult handleCommand(
            std::string_view command,
            std::span<std::string_view> args,
            util::Instant startTime
        ) = 0;

        // engine -> gui
        virtual void printSearchInfo(std::ostream& stream, const SearchInfo& info) const = 0;
        virtual void printInfoString(std::ostream& stream, std::string_view str) const = 0;
        virtual void printBestMove(std::ostream& stream, Move move) const = 0;
        virtual void handleNoLegalMoves() const = 0;
        virtual bool handleEnteringKingsWin() const = 0;
    };

    constexpr std::string_view kDefaultHandler = "usi";

    [[nodiscard]] std::unique_ptr<IProtocolHandler> createHandler(std::string_view name, EngineState& state);

    [[nodiscard]] const IProtocolHandler& currHandler();
} // namespace stoat::protocol
