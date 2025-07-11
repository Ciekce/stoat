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

#include "eval.h"

#include <algorithm>

namespace stoat::eval {
    Score staticEval(const Position& pos, const nnue::NnueState& nnueState) {
        const auto nnue = nnueState.evaluate(pos.stm());
        return std::clamp(nnue, -kScoreWin + 1, kScoreWin - 1);
    }

    Score staticEvalOnce(const Position& pos) {
        const auto nnue = nnue::evaluateOnce(pos);
        return std::clamp(nnue, -kScoreWin + 1, kScoreWin - 1);
    }

    Score correctedStaticEval(
        const Position& pos,
        const nnue::NnueState& nnueState,
        const CorrectionHistoryTable& corrhist
    ) {
        const auto eval = staticEval(pos, nnueState);
        const auto correction = corrhist.correction(pos);
        return std::clamp(eval + correction, -kScoreWin + 1, kScoreWin - 1);
    }
} // namespace stoat::eval
