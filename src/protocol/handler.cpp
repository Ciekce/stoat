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

#include "handler.h"

#include <functional>

#include "../util/string_map.h"
#include "uci.h"
#include "usi.h"

namespace stoat::protocol {
    namespace {
        using HandlerFactory = std::function<std::unique_ptr<IProtocolHandler>(EngineState&)>;

        const auto s_factories = [] {
            util::UnorderedStringMap<HandlerFactory> map{};

            const auto registerHandler = [&map]<typename HandlerType>(std::string_view name) {
                map[std::string{name}] = [](EngineState& state) { return std::make_unique<HandlerType>(state); };
            };

            registerHandler.template operator()<UsiHandler>("usi");
            registerHandler.template operator()<UciHandler>("uci");

            return map;
        }();
    } // namespace

    std::unique_ptr<IProtocolHandler> createHandler(std::string_view name, EngineState& state) {
        if (const auto itr = s_factories.find(name); itr != s_factories.end()) {
            return itr->second(state);
        }

        return nullptr;
    }
} // namespace stoat::protocol
