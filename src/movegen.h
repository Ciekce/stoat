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

#include "types.h"

#include "move.h"
#include "position.h"
#include "util/static_vector.h"

namespace stoat::movegen {
    using MoveList = util::StaticVector<Move, 600>;

    void generateAll(MoveList& dst, const Position& pos);
    void generateCaptures(MoveList& dst, const Position& pos);
    void generateNonCaptures(MoveList& dst, const Position& pos);
    void generateRecaptures(MoveList& dst, const Position& pos, Square captureSq);
} // namespace stoat::movegen
