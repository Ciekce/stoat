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

#include "usi.h"

#include <algorithm>
#include <cassert>

namespace stoat::protocol {
    UsiHandler::UsiHandler(EngineState& state) :
            UciLikeHandler{state} {
        registerCommandHandler("usinewgame", [this](std::span<std::string_view>, util::Instant) { handleNewGame(); });
    }

    void UsiHandler::printOptionName(std::ostream& stream, std::string_view name) const {
        static constexpr std::array kFixedSemanticsOptions = {
            "Hash",
        };

        if (std::ranges::find(kFixedSemanticsOptions, name) != kFixedSemanticsOptions.end()) {
            stream << "USI_";
        }

        stream << name;
    }

    void UsiHandler::finishInitialInfo() const {
        std::cout << "usiok" << std::endl;
    }

    util::Result<Position, std::optional<std::string>> UsiHandler::parsePosition(std::span<std::string_view> args) {
        assert(!args.empty());

        if (args[0] != "sfen") {
            return util::err<std::optional<std::string>>();
        }

        if (args.size() == 1) {
            return util::err<std::optional<std::string>>("Missing sfen");
        }

        return Position::fromSfenParts(args.subspan<1>()).mapErr<std::optional<std::string>>([](const SfenError& err) {
            return std::optional{"Failed to parse sfen: " + std::string{err.message()}};
        });
    }

    util::Result<Move, InvalidMoveError> UsiHandler::parseMove(std::string_view str) {
        return Move::fromStr(str);
    }

    void UsiHandler::printBoard(std::ostream& stream, const Position& pos) const {
        stream << pos;
    }

    void UsiHandler::printFen(std::ostream& stream, const Position& pos) const {
        stream << pos.sfen();
    }

    void UsiHandler::printMove(std::ostream& stream, Move move) const {
        stream << move;
    }

    void UsiHandler::printMateScore(std::ostream& stream, i32 plies) const {
        stream << plies;
    }

    void UsiHandler::printFenLine(std::ostream& stream, const Position& pos) const {
        stream << "Sfen: " << pos.sfen() << '\n';
    }
} // namespace stoat::protocol
