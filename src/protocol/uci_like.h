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

#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "../util/result.h"
#include "../util/string_map.h"
#include "../util/timer.h"
#include "handler.h"

#include "../util/rng.h"

namespace stoat::protocol {
    class UciLikeHandler : public IProtocolHandler {
    public:
        explicit UciLikeHandler(EngineState& state);
        ~UciLikeHandler() override = default;

        void printInitialInfo() const final;

        [[nodiscard]] CommandResult handleCommand(
            std::string_view command,
            std::span<std::string_view> args,
            util::Instant startTime
        ) final;

        void printSearchInfo(std::ostream& stream, const SearchInfo& info) const final;
        void printInfoString(std::ostream& stream, std::string_view str) const final;
        void printBestMove(std::ostream& stream, Move move) const final;

    protected:
        using CommandHandlerType = std::function<void(std::span<std::string_view>, util::Instant)>;
        void registerCommandHandler(std::string_view command, CommandHandlerType handler);

        void handleNewGame();

        virtual void printOptionName(std::ostream& stream, std::string_view name) const = 0;
        [[nodiscard]] virtual std::string transformOptionName(std::string_view name) const = 0;

        virtual void finishInitialInfo() const = 0;

        [[nodiscard]] virtual util::Result<Position, std::optional<std::string>> parsePosition(
            std::span<std::string_view> args
        ) const = 0;
        [[nodiscard]] virtual util::Result<Move, InvalidMoveError> parseMove(std::string_view str) const = 0;

        virtual void printBoard(std::ostream& stream, const Position& pos) const = 0;
        virtual void printFen(std::ostream& stream, const Position& pos) const = 0;
        virtual void printMove(std::ostream& stream, Move move) const = 0;
        virtual void printMateScore(std::ostream& stream, i32 plies) const = 0;

        // ech
        virtual void printFenLine(std::ostream& stream, const Position& pos) const = 0;

        [[nodiscard]] virtual std::string_view btimeToken() const = 0;
        [[nodiscard]] virtual std::string_view wtimeToken() const = 0;

        [[nodiscard]] virtual std::string_view bincToken() const = 0;
        [[nodiscard]] virtual std::string_view wincToken() const = 0;

    private:
        util::UnorderedStringMap<CommandHandlerType> m_cmdHandlers{};

        EngineState& m_state;

        void handle_isready(std::span<std::string_view> args, util::Instant startTime);
        void handle_position(std::span<std::string_view> args, util::Instant startTime);
        void handle_go(std::span<std::string_view> args, util::Instant startTime);
        void handle_stop(std::span<std::string_view> args, util::Instant startTime);
        void handle_setoption(std::span<std::string_view> args, util::Instant startTime);

        // nonstandard
        void handle_d(std::span<std::string_view> args, util::Instant startTime);
        void handle_splitperft(std::span<std::string_view> args, util::Instant startTime);
    };
} // namespace stoat::protocol
