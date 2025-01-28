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

#include "uci_like.h"

namespace stoat::protocol {
    class UsiHandler : public UciLikeHandler {
    public:
        explicit UsiHandler(EngineState& state);
        ~UsiHandler() override = default;

        void printOptionName(std::ostream& stream, std::string_view name) const final;
        [[nodiscard]] std::string transformOptionName(std::string_view name) const final;

        void finishInitialInfo() const final;

        [[nodiscard]] util::Result<Position, std::optional<std::string>> parsePosition(std::span<std::string_view> args
        ) const final;
        [[nodiscard]] util::Result<Move, InvalidMoveError> parseMove(std::string_view str) const final;

        void printBoard(std::ostream& stream, const Position& pos) const final;
        void printFen(std::ostream& stream, const Position& pos) const final;
        void printMove(std::ostream& stream, Move move) const final;
        void printMateScore(std::ostream& stream, i32 plies) const final;

        void printFenLine(std::ostream& stream, const Position& pos) const final;

        [[nodiscard]] std::string_view btimeToken() const final;
        [[nodiscard]] std::string_view wtimeToken() const final;

        [[nodiscard]] std::string_view bincToken() const final;
        [[nodiscard]] std::string_view wincToken() const final;
    };
} // namespace stoat::protocol
