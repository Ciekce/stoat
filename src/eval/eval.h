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

#include "../core.h"
#include "../correction.h"
#include "../position.h"
#include "nnue.h"

namespace stoat::eval {
    [[nodiscard]] Score staticEval(const Position& pos, const nnue::NnueState& nnueState);
    [[nodiscard]] Score staticEvalOnce(const Position& pos);

    [[nodiscard]] Score adjustEval(Score rawEval, const Position& pos, const CorrectionHistory& corrhist, i32 ply);

    [[nodiscard]] Score adjustedEval(
        const Position& pos,
        const nnue::NnueState& nnueState,
        const CorrectionHistory& corrhist,
        i32 ply
    );
} // namespace stoat::eval
