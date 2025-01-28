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

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "bench.h"
#include "protocol/handler.h"
#include "util/split.h"

using namespace stoat;

namespace {
    // :doom:
    const protocol::IProtocolHandler* s_currHandler;
} // namespace

namespace stoat::protocol {
    const IProtocolHandler& currHandler() {
        return *s_currHandler;
    }
} // namespace stoat::protocol

i32 main(i32 argc, char* argv[]) {
    protocol::EngineState state{};

    std::string currHandler{protocol::kDefaultHandler};
    auto handler = protocol::createHandler(currHandler, state);

    s_currHandler = handler.get();

    if (argc > 1) {
        const auto subcommand = std::string_view{argv[1]};
        if (subcommand == "bench") {
            bench::run();
            return 0;
        }
    }

    // *must* be destroyed before the handler
    Searcher searcher{tt::kDefaultTtSizeMib};
    state.searcher = &searcher;

    std::vector<std::string_view> tokens{};

    std::string line{};
    while (std::getline(std::cin, line)) {
        const auto startTime = util::Instant::now();

        tokens.clear();
        util::split(tokens, line);

        if (tokens.empty()) {
            continue;
        }

        const auto command = tokens[0];
        const auto args = std::span{tokens}.subspan<1>();

        if (command == currHandler) {
            handler->printInitialInfo();
            continue;
        }

        const auto result = handler->handleCommand(command, args, startTime);

        if (result == protocol::CommandResult::kQuit) {
            break;
        } else if (result == protocol::CommandResult::kUnknown) {
            if (auto newHandler = protocol::createHandler(command, state)) {
                if (searcher.isSearching()) {
                    std::cerr << "Still searching" << std::endl;
                    continue;
                }

                currHandler = command;
                handler = std::move(newHandler);

                s_currHandler = handler.get();

                handler->printInitialInfo();
            } else {
                std::cerr << "Unknown command '" << command << "'" << std::endl;
            }
        }
    }
}
