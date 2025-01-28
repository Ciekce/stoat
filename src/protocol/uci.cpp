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

#include "uci.h"

#include <cassert>
#include <sstream>
#include <vector>

#include "../util/parse.h"
#include "../util/split.h"

// UCI adapter for cutechess
// messy

namespace stoat::protocol {
    namespace {
        [[nodiscard]] Square parseSquare(std::string_view str) {
            if (str.length() != 2) {
                return Squares::kNone;
            }

            if (str[0] < 'a' || str[0] > 'i' || str[1] < '1' || str[1] > '9') {
                return Squares::kNone;
            }

            const u32 file = str[0] - 'a';
            const u32 rank = str[1] - '1';

            return Square::fromRaw(rank * 9 + file);
        }

        void printSquare(std::ostream& stream, Square sq) {
            assert(sq != Squares::kNone);

            stream << static_cast<char>('a' + sq.file());
            stream << static_cast<char>('1' + sq.rank());
        }
    } // namespace

    UciHandler::UciHandler(EngineState& state) :
            UciLikeHandler{state} {
        registerCommandHandler("ucinewgame", [this](std::span<std::string_view>, util::Instant) { handleNewGame(); });
    }

    void UciHandler::printOptionName(std::ostream& stream, std::string_view name) const {
        stream << name;
    }

    std::string UciHandler::transformOptionName(std::string_view name) const {
        if (name.starts_with("uci_")) {
            return std::string{name.substr(4)};
        }

        return std::string{name};
    }

    void UciHandler::finishInitialInfo() const {
        std::cout << "option name UCI_Variant type combo default shogi var shogi\n";
        std::cout << "\ninfo string Stoat's UCI support is intended for Cute Chess compatibility only.\n";
        std::cout << "info string Prefer USI for normal use.\n";
        std::cout << "uciok" << std::endl;
    }

    util::Result<Position, std::optional<std::string>> UciHandler::parsePosition(std::span<std::string_view> args
    ) const {
        assert(!args.empty());

        if (args[0] != "fen") {
            return util::err<std::optional<std::string>>();
        }

        if (args.size() == 1) {
            return util::err<std::optional<std::string>>("Missing fen");
        }

        if (args.size() < 4 || args.size() > 5) {
            return util::err<std::optional<std::string>>("Failed to parse FEN: wrong number of FEN parts");
        }

        std::ostringstream sfen{};

        const auto handStart = args[1].find_first_of('[');

        if (handStart == 0) {
            return util::err<std::optional<std::string>>("Failed to parse FEN: missing board");
        }

        if (handStart == std::string_view::npos) {
            return util::err<std::optional<std::string>>("Failed to parse FEN: failed to find hand");
        }

        const auto handEnd = args[1].find_first_of(']', handStart + 1);

        if (handEnd == std::string_view::npos) {
            return util::err<std::optional<std::string>>("Failed to parse FEN: failed to find hand");
        }

        if (args[2] != "w" && args[2] != "b") {
            return util::err<std::optional<std::string>>("Failed to parse FEN: invalid side to move");
        }

        const auto board = args[1].substr(0, handStart);
        const auto hand =
            handStart == handEnd - 1 ? std::string_view{"-"} : args[1].substr(handStart + 1, handEnd - handStart - 1);
        const auto stm = args[2] == "w" ? 'b' : 'w';

        sfen << board << ' ' << stm << ' ' << hand;

        if (args.size() == 5) {
            if (const auto fullmove = util::tryParse<u32>(args[4])) {
                const auto moveCount = *fullmove * 2 - (stm == 'b');
                sfen << ' ' << moveCount;
            } else {
                return util::err<std::optional<std::string>>("Failed to parse FEN: invalid fullmove number");
            }
        }

        std::cout << "info string constructed sfen: " << sfen.view() << std::endl;

        return Position::fromSfen(sfen.view()).mapErr<std::optional<std::string>>([](const SfenError& err) {
            return std::optional{"Failed to parse constructed sfen: " + std::string{err.message()}};
        });
    }

    util::Result<Move, InvalidMoveError> UciHandler::parseMove(std::string_view str) const {
        if (str.size() < 4 || str.size() > 5) {
            return util::err<InvalidMoveError>();
        }

        if (str[1] == '@') {
            if (str.size() != 4) {
                return util::err<InvalidMoveError>();
            }

            const auto piece = PieceType::unpromotedFromChar(str[0]);
            const auto square = parseSquare(str.substr(2, 2));

            if (!piece || !square || piece == PieceTypes::kKing) {
                return util::err<InvalidMoveError>();
            }

            return util::ok(Move::makeDrop(piece, square));
        }

        if (str.size() == 5 && str[4] != '+') {
            return util::err<InvalidMoveError>();
        }

        const bool promo = str.size() == 5;

        const auto from = parseSquare(str.substr(0, 2));
        const auto to = parseSquare(str.substr(2, 2));

        if (!from || !to) {
            return util::err<InvalidMoveError>();
        }

        return util::ok(promo ? Move::makePromotion(from, to) : Move::makeNormal(from, to));
    }

    void UciHandler::printBoard(std::ostream& stream, const Position& pos) const {
        stream << " +---+---+---+---+---+---+---+---+---+\n";

        for (i32 rank = 8; rank >= 0; --rank) {
            for (i32 file = 0; file < 9; ++file) {
                const auto piece = pos.pieceOn(Square::fromFileRank(file, rank));

                if (piece) {
                    stream << " |" << (!piece.type().isPromoted() ? " " : "") << piece;
                } else {
                    stream << " |  ";
                }
            }

            stream << " | " << static_cast<char>('1' + rank);
            stream << "\n +---+---+---+---+---+---+---+---+---+\n";
        }

        stream << "   a   b   c   d   e   f   g   h   i\n";

        stream << "\nBlack pieces in hand: " << pos.hand(Colors::kBlack);
        stream << "\nWhite pieces in hand: " << pos.hand(Colors::kWhite);

        stream << "\n\n" << (pos.stm() == Colors::kBlack ? "Black" : "White") << " to move";
    }

    void UciHandler::printFen(std::ostream& stream, const Position& pos) const {
        const auto sfen = pos.sfen();

        std::vector<std::string_view> split{};
        split.reserve(4);

        util::split(split, sfen);
        assert(split.size() == 4);

        const auto stm = split[1] == "w" ? 'b' : 'w';
        const auto fullmove = (pos.moveCount() + 1) / 2;

        stream << split[0] << '[' << split[2] << "] " << stm << " - - 0 " << fullmove;
    }

    void UciHandler::printMove(std::ostream& stream, Move move) const {
        if (move.isNull()) {
            stream << "0000";
            return;
        }

        if (move.isDrop()) {
            const auto square = move.to();
            const auto piece = move.dropPiece();

            stream << piece.str()[0] << '@';
            printSquare(stream, square);

            return;
        }

        const auto to = move.to();
        const auto from = move.from();

        printSquare(stream, from);
        printSquare(stream, to);

        if (move.isPromo()) {
            stream << '+';
        }
    }

    void UciHandler::printMateScore(std::ostream& stream, i32 plies) const {
        if (plies > 0) {
            stream << ((plies + 1) / 2);
        } else {
            stream << (plies / 2);
        }
    }

    void UciHandler::printFenLine(std::ostream& stream, const Position& pos) const {
        stream << "Fen: ";
        printFen(stream, pos);
        stream << '\n';
    }

    std::string_view UciHandler::btimeToken() const {
        return "wtime";
    }

    std::string_view UciHandler::wtimeToken() const {
        return "btime";
    }

    std::string_view UciHandler::bincToken() const {
        return "winc";
    }

    std::string_view UciHandler::wincToken() const {
        return "binc";
    }
} // namespace stoat::protocol
