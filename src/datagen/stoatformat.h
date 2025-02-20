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

#include <array>
#include <utility>
#include <vector>

#include "../util/u4array.h"
#include "format.h"

namespace stoat::datagen::format {
    struct __attribute__((packed)) StoatformatRecord {
        std::array<u128, 2> occ{};
        util::U4Array<40> pieces{};
        i16 score{};
        u16 plyCount{};
        [[maybe_unused]] std::array<std::byte, 8> _unused{};

        [[nodiscard]] Color stm() const;
        void setStm(Color stm);

        [[nodiscard]] Outcome wdl() const;
        void setWdl(Outcome wdl);

        [[nodiscard]] static StoatformatRecord pack(const Position& pos, i16 senteScore, Outcome wdl);
    };

    static_assert(sizeof(StoatformatRecord) == 64);
} // namespace stoat::datagen::format
