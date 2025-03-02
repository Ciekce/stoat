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

#include "uci_like.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../eval/eval.h"
#include "../eval/nnue.h"
#include "../limit.h"
#include "../perft.h"
#include "../ttable.h"
#include "../util/parse.h"
#include "common.h"

namespace stoat::protocol {
    namespace {
        struct PrintScore {
            Score value;

            inline friend std::ostream& operator<<(std::ostream& stream, PrintScore score) {
                if (score.value > 0) {
                    stream << '+';
                } else if (score.value < 0) {
                    stream << '-';
                }

                const auto absScore = std::abs(score.value);

                stream << (absScore / 100);
                stream << '.';
                stream << std::setw(2) << std::setfill('0') << (absScore % 100);

                return stream;
            }
        };
    } // namespace

    UciLikeHandler::UciLikeHandler(EngineState& state) :
            m_state{state} {
#define REGISTER_HANDLER(Command) \
    registerCommandHandler(#Command, [this](auto args, auto startTime) { handle_##Command(args, startTime); })

        REGISTER_HANDLER(position);
        REGISTER_HANDLER(go);
        REGISTER_HANDLER(stop);
        REGISTER_HANDLER(setoption);

        REGISTER_HANDLER(d);
        REGISTER_HANDLER(splitperft);
        REGISTER_HANDLER(raweval);

#undef REGISTER_HANDLER
    }

    void UciLikeHandler::printInitialInfo() const {
        std::cout << "id name " << kName << ' ' << kVersion << '\n';
        std::cout << "id author " << kAuthor << '\n';

        // dummy options for OB
        std::cout << "option name ";
        printOptionName(std::cout, "Hash");
        std::cout << " type spin default " << tt::kDefaultTtSizeMib << " min " << tt::kTtSizeRange.min() << " max "
                  << tt::kTtSizeRange.max() << '\n';

        std::cout << "option name ";
        printOptionName(std::cout, "Threads");
        std::cout << " type spin default 1 min 1 max 1\n";

        std::cout << "option name ";
        printOptionName(std::cout, "CuteChessWorkaround");
        std::cout << " type check default false\n";

        finishInitialInfo();
    }

    CommandResult UciLikeHandler::handleCommand(
        std::string_view command,
        std::span<std::string_view> args,
        util::Instant startTime
    ) {
        if (command == "quit") {
            return CommandResult::kQuit;
        }

        if (auto itr = m_cmdHandlers.find(command); itr != m_cmdHandlers.end()) {
            itr->second(args, startTime);
            return CommandResult::kContinue;
        }

        return CommandResult::kUnknown;
    }

    void UciLikeHandler::printSearchInfo(std::ostream& stream, const SearchInfo& info) const {
        stream << "info depth " << info.depth;

        if (info.seldepth) {
            stream << " seldepth " << *info.seldepth;
        }

        if (info.timeSec) {
            const auto ms = static_cast<usize>(*info.timeSec * 1000.0);
            stream << " time " << ms;
        }

        stream << " nodes " << info.nodes;

        if (info.timeSec) {
            const auto nps = static_cast<usize>(static_cast<f64>(info.nodes) / *info.timeSec);
            stream << " nps " << nps;
        }

        stream << " score ";

        if (std::holds_alternative<MateDisplayScore>(info.score)) {
            const auto plies = std::get<MateDisplayScore>(info.score).plies;
            stream << "mate ";
            printMateScore(std::cout, plies);
        } else {
            const auto score = std::get<CpDisplayScore>(info.score).score;
            stream << "cp " << score;
        }

        if (info.hashfull) {
            stream << " hashfull " << *info.hashfull;
        }

        stream << " pv";

        for (usize i = 0; i < info.pv.length; ++i) {
            stream << ' ';
            printMove(stream, info.pv.moves[i]);
        }

        stream << std::endl;
    }

    void UciLikeHandler::printInfoString(std::ostream& stream, std::string_view str) const {
        stream << "info string " << str << std::endl;
    }

    void UciLikeHandler::printBestMove(std::ostream& stream, Move move) const {
        stream << "bestmove ";
        printMove(stream, move);
        stream << std::endl;
    }

    void UciLikeHandler::registerCommandHandler(std::string_view command, CommandHandlerType handler) {
        if (m_cmdHandlers.contains(command)) {
            std::cerr << "tried to overwrite command handler for '" << command << "'" << std::endl;
            return;
        }

        m_cmdHandlers[std::string{command}] = std::move(handler);
    }

    void UciLikeHandler::handleNewGame() {
        if (m_state.searcher->isSearching()) {
            std::cerr << "Still searching" << std::endl;
            return;
        }

        m_state.searcher->newGame();
    }

    void UciLikeHandler::handle_position(std::span<std::string_view> args, [[maybe_unused]] util::Instant startTime) {
        if (m_state.searcher->isSearching()) {
            std::cerr << "Still searching" << std::endl;
            return;
        }

        if (args.empty()) {
            return;
        }

        usize next = 0;

        if (args[0] == "startpos") {
            m_state.pos = Position::startpos();
            m_state.keyHistory.clear();

            next = 1;
        } else {
            const auto count = std::distance(args.begin(), std::ranges::find(args, "moves"));
            if (auto parsed = parsePosition(args.subspan(0, count))) {
                m_state.pos = parsed.take();
                m_state.keyHistory.clear();
            } else {
                if (const auto err = parsed.takeErr()) {
                    std::cerr << *err << std::endl;
                }
                return;
            }

            next = count;
        }

        assert(next <= args.size());

        if (next >= args.size() || args[next] != "moves") {
            return;
        }

        for (usize i = next + 1; i < args.size(); ++i) {
            if (auto parsedMove = parseMove(args[i])) {
                m_state.keyHistory.push_back(m_state.pos.key());
                m_state.pos = m_state.pos.applyMove(parsedMove.take());
            } else {
                std::cerr << "Invalid move '" << args[i] << "'" << std::endl;
                break;
            }
        }
    }

    void UciLikeHandler::handle_go(std::span<std::string_view> args, util::Instant startTime) {
        if (m_state.searcher->isSearching()) {
            std::cerr << "Still searching" << std::endl;
            return;
        }

        auto limiter = std::make_unique<limit::CompoundLimiter>();

        bool infinite = false;

        auto maxDepth = kMaxDepth;

        std::optional<f64> btime{};
        std::optional<f64> wtime{};

        std::optional<f64> binc{};
        std::optional<f64> winc{};

        std::optional<f64> byoyomi{};

        for (i32 i = 0; i < args.size(); ++i) {
            if (args[i] == "infinite") {
                infinite = true;
            } else if (args[i] == "depth") {
                if (++i == args.size()) {
                    std::cerr << "Missing depth" << std::endl;
                    return;
                }

                if (!util::tryParse(maxDepth, args[i])) {
                    std::cerr << "Invalid depth '" << args[i] << "'" << std::endl;
                    return;
                }
            } else if (args[i] == "nodes") {
                if (++i == args.size()) {
                    std::cerr << "Missing node limit" << std::endl;
                    return;
                }

                usize maxNodes{};

                if (!util::tryParse(maxNodes, args[i])) {
                    std::cerr << "Invalid node limit '" << args[i] << "'" << std::endl;
                    return;
                }

                limiter->addLimiter<limit::NodeLimiter>(maxNodes);
            } else if (args[i] == "movetime") {
                if (++i == args.size()) {
                    std::cerr << "Missing move time limit" << std::endl;
                    return;
                }

                i64 maxTimeMs{};

                if (!util::tryParse(maxTimeMs, args[i])) {
                    std::cerr << "Invalid move time limit '" << args[i] << "'" << std::endl;
                    return;
                }

                maxTimeMs = std::max<i64>(maxTimeMs, 1);

                const auto maxTimeSec = static_cast<f64>(maxTimeMs) / 1000.0;
                limiter->addLimiter<limit::MoveTimeLimiter>(startTime, maxTimeSec);
            } else if (args[i] == btimeToken()) {
                if (++i == args.size()) {
                    std::cerr << "Missing " << btimeToken() << " limit" << std::endl;
                    return;
                }

                i64 btimeMs{};

                if (!util::tryParse(btimeMs, args[i])) {
                    std::cerr << "Invalid " << btimeToken() << " limit '" << args[i] << "'" << std::endl;
                    return;
                }

                btimeMs = std::max<i64>(btimeMs, 1);
                btime = static_cast<f64>(btimeMs) / 1000.0;
            } else if (args[i] == wtimeToken()) {
                if (++i == args.size()) {
                    std::cerr << "Missing " << wtimeToken() << " limit" << std::endl;
                    return;
                }

                i64 wtimeMs{};

                if (!util::tryParse(wtimeMs, args[i])) {
                    std::cerr << "Invalid " << wtimeToken() << " limit '" << args[i] << "'" << std::endl;
                    return;
                }

                wtimeMs = std::max<i64>(wtimeMs, 1);
                wtime = static_cast<f64>(wtimeMs) / 1000.0;
            } else if (args[i] == bincToken()) {
                if (++i == args.size()) {
                    std::cerr << "Missing " << bincToken() << " limit" << std::endl;
                    return;
                }

                i64 bincMs{};

                if (!util::tryParse(bincMs, args[i])) {
                    std::cerr << "Invalid " << bincToken() << " limit '" << args[i] << "'" << std::endl;
                    return;
                }

                bincMs = std::max<i64>(bincMs, 0);
                binc = static_cast<f64>(bincMs) / 1000.0;
            } else if (args[i] == wincToken()) {
                if (++i == args.size()) {
                    std::cerr << "Missing " << wincToken() << " limit" << std::endl;
                    return;
                }

                i64 wincMs{};

                if (!util::tryParse(wincMs, args[i])) {
                    std::cerr << "Invalid " << wincToken() << " limit '" << args[i] << "'" << std::endl;
                    return;
                }

                wincMs = std::max<i64>(wincMs, 0);
                winc = static_cast<f64>(wincMs) / 1000.0;
            } else if (args[i] == "byoyomi") {
                if (++i == args.size()) {
                    std::cerr << "Missing byoyomi" << std::endl;
                    return;
                }

                i64 byoyomiMs{};

                if (!util::tryParse(byoyomiMs, args[i])) {
                    std::cerr << "Invalid byoyomi '" << args[i] << "'" << std::endl;
                    return;
                }

                byoyomiMs = std::max<i64>(byoyomiMs, 0);
                byoyomi = static_cast<f64>(byoyomiMs) / 1000.0;
            } else if (args[i] == "mate") {
                printInfoString(std::cout, "go mate not supported");
                printGoMateResponse(std::cout);
                return;
            }
        }

        const auto time = m_state.pos.stm() == Colors::kBlack ? btime : wtime;
        const auto inc = m_state.pos.stm() == Colors::kBlack ? binc : winc;

        if (time) {
            const limit::TimeLimits limits{
                .remaining = *time,
                .increment = inc.value_or(0.0),
                .byoyomi = byoyomi.value_or(0.0),
            };

            limiter->addLimiter<limit::TimeManager>(startTime, limits);
        } else if (inc) {
            printInfoString(std::cout, "Warning: increment given but no time, ignoring");
        }

        m_state.searcher
            ->startSearch(m_state.pos, m_state.keyHistory, startTime, infinite, maxDepth, std::move(limiter));
    }

    void UciLikeHandler::handle_stop(std::span<std::string_view> args, [[maybe_unused]] util::Instant startTime) {
        if (m_state.searcher->isSearching()) {
            m_state.searcher->stop();
        } else {
            std::cerr << "Not searching" << std::endl;
        }
    }

    void UciLikeHandler::handle_setoption(std::span<std::string_view> args, [[maybe_unused]] util::Instant startTime) {
        if (m_state.searcher->isSearching()) {
            std::cerr << "Still searching" << std::endl;
            return;
        }

        if (args.size() < 2 || args[0] != "name") {
            return;
        }

        //TODO handle options more generically

        const auto valueIdx = std::distance(args.begin(), std::ranges::find(args, "value"));

        if (valueIdx == 1) {
            std::cerr << "Missing option name" << std::endl;
            return;
        }

        if (valueIdx >= args.size() - 1) {
            std::cerr << "Missing value" << std::endl;
            return;
        }

        if (valueIdx > 2) {
            std::ostringstream str{};

            bool first = true;
            for (usize i = 2; i < valueIdx; ++i) {
                if (!first) {
                    str << ' ';
                } else {
                    first = false;
                }
                str << args[i];
            }

            printInfoString(std::cout, "Warning: spaces in option names not supported, skipping \"" + str.str() + "\"");
        }

        std::string name{};
        name.reserve(args[1].length());
        std::ranges::transform(args[1], std::back_inserter(name), [](char c) {
            return static_cast<char>(std::tolower(c));
        });

        name = transformOptionName(name);

        std::ostringstream valueStr{};

        bool first = true;
        for (usize i = valueIdx + 1; i < args.size(); ++i) {
            if (!first) {
                valueStr << ' ';
            } else {
                first = false;
            }
            valueStr << args[i];
        }

        const auto value = valueStr.view();
        assert(!value.empty());

        if (name == "hash") {
            if (const auto newHash = util::tryParse<usize>(value)) {
                const auto size = tt::kTtSizeRange.clamp(*newHash);
                m_state.searcher->setTtSize(size);
            } else {
                std::cerr << "Invalid hash size '" << value << "'" << std::endl;
            }
        } else if (name == "threads") {
            //
        } else if (name == "cutechessworkaround") {
            if (const auto newCcWorkaround = util::tryParseBool(value)) {
                m_state.searcher->setCuteChessWorkaround(*newCcWorkaround);
            } else {
                std::cerr << "Invalid check value '" << value << "'" << std::endl;
            }
        } else {
            std::cerr << "Unknown option '" << args[1] << "'" << std::endl;
        }
    }

    void UciLikeHandler::handle_d(
        [[maybe_unused]] std::span<std::string_view> args,
        [[maybe_unused]] util::Instant startTime
    ) {
        const auto printKey = [](u64 key) {
            std::ostringstream str{};
            str << "0x" << std::hex << std::setw(16) << std::setfill('0') << key;
            std::cout << str.view();
        };

        std::cout << '\n';
        printBoard(std::cout, m_state.pos);

        std::cout << "\n\n";
        printFenLine(std::cout, m_state.pos);

        std::cout << "Key: ";
        printKey(m_state.pos.key());

        std::cout << "\nCheckers:";

        auto checkers = m_state.pos.checkers();
        while (!checkers.empty()) {
            std::cout << ' ' << checkers.popLsb();
        }

        std::cout << "\nPinned:";

        auto pinned = m_state.pos.pinned();
        while (!pinned.empty()) {
            std::cout << ' ' << pinned.popLsb();
        }

        std::cout << "\nStatic eval: ";

        const auto staticEval = eval::staticEval(m_state.pos);
        std::cout << PrintScore{staticEval};

        std::cout << std::endl;
    }

    void UciLikeHandler::handle_splitperft(std::span<std::string_view> args, [[maybe_unused]] util::Instant startTime) {
        if (args.empty()) {
            return;
        }

        if (const auto depth = util::tryParse<i32>(args[0])) {
            splitPerft(m_state.pos, *depth);
        }
    }

    void UciLikeHandler::handle_raweval(
        [[maybe_unused]] std::span<std::string_view> args,
        [[maybe_unused]] util::Instant startTime
    ) {
        std::cout << eval::nnue::evaluate(m_state.pos) << std::endl;
    }
} // namespace stoat::protocol
