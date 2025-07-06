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

#include "../../types.h"

#include <iostream>

#include "../../core.h"
#include "../../move.h"
#include "../../position.h"

namespace stoat::datagen::format {
    enum class Outcome : u8 {
        kBlackLoss = 0,
        kDraw,
        kBlackWin,
    };

    class IDataFormat {
    public:
        virtual ~IDataFormat() = default;

        virtual void startStandard() = 0;
        virtual void start(const Position& pos) = 0;

        virtual void pushUnscored(Move move) = 0;
        virtual void push(Move move, Score score) = 0;

        virtual usize writeAllWithOutcome(std::ostream& stream, Outcome outcome) = 0;
    };
} // namespace stoat::datagen::format
