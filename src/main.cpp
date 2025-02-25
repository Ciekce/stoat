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
#include "datagen/datagen.h"
#include "protocol/handler.h"
#include "util/ctrlc.h"
#include "util/parse.h"
#include "util/split.h"

namespace stoat {
    namespace {
        i32 runDatagen(std::span<const std::string_view> args) {
            const auto printUsage = [&] {
                std::cerr << "usage: " << args[0] << " datagen <path> [threads]" << std::endl;
            };

            if (args.size() < 3) {
                printUsage();
                return 1;
            }

            u32 threads = 1;

            if (args.size() >= 4 && !util::tryParse(threads, args[3])) {
                std::cerr << "invalid thread count \"" << args[3] << "\"" << std::endl;
                printUsage();
                return 1;
            }

            return datagen::run(args[2], threads);
        }

        // :doom:
        const protocol::IProtocolHandler* s_currHandler;
    } // namespace

    namespace protocol {
        const IProtocolHandler& currHandler() {
            return *s_currHandler;
        }
    } // namespace protocol

    i32 main(std::span<const std::string_view> args) {
        util::signal::init();

        protocol::EngineState state{};

        std::string currHandler{protocol::kDefaultHandler};
        auto handler = protocol::createHandler(currHandler, state);

        s_currHandler = handler.get();

        if (args.size() > 1) {
            const auto subcommand = args[1];
            if (subcommand == "bench") {
                bench::run();
                return 0;
            } else if (subcommand == "datagen") {
                return runDatagen(args);
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
            const auto commandArgs = std::span{tokens}.subspan<1>();

            if (command == currHandler) {
                handler->printInitialInfo();
                continue;
            }

            const auto result = handler->handleCommand(command, commandArgs, startTime);

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

        return 0;
    }
} // namespace stoat

using namespace stoat;

i32 main(i32 argc, char* argv[]) {
    std::vector<std::string_view> args{};
    args.reserve(argc);

    for (i32 i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    return main(args);
}
