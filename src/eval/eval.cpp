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
#include <cmath>

#include "../attacks/attacks.h"
#include "material.h"

namespace stoat::eval {
    namespace {
        constexpr Score kKingRingPieceScale = 8;

        [[nodiscard]] Score evalMaterial(const Position& pos, Color c) {
            const auto materialCount = [&](PieceType pt) {
                const auto count = pos.pieceBb(pt, c).popcount();
                return count * pieceValue(pt);
            };

            Score score{};

            score += materialCount(PieceTypes::kPawn);
            score += materialCount(PieceTypes::kPromotedPawn);
            score += materialCount(PieceTypes::kLance);
            score += materialCount(PieceTypes::kKnight);
            score += materialCount(PieceTypes::kPromotedLance);
            score += materialCount(PieceTypes::kPromotedKnight);
            score += materialCount(PieceTypes::kSilver);
            score += materialCount(PieceTypes::kPromotedSilver);
            score += materialCount(PieceTypes::kGold);
            score += materialCount(PieceTypes::kBishop);
            score += materialCount(PieceTypes::kRook);
            score += materialCount(PieceTypes::kPromotedBishop);
            score += materialCount(PieceTypes::kPromotedRook);

            const auto& hand = pos.hand(c);

            if (hand.empty()) {
                return score;
            }

            const auto handPieceValue = [&](PieceType pt) { return static_cast<i32>(hand.count(pt)) * pieceValue(pt); };

            score += handPieceValue(PieceTypes::kPawn);
            score += handPieceValue(PieceTypes::kLance);
            score += handPieceValue(PieceTypes::kKnight);
            score += handPieceValue(PieceTypes::kSilver);
            score += handPieceValue(PieceTypes::kGold);
            score += handPieceValue(PieceTypes::kBishop);
            score += handPieceValue(PieceTypes::kRook);

            return score;
        }

        [[nodiscard]] Score evalKingSafety(const Position& pos, Color c) {
            const auto stmPieces = pos.colorBb(c);
            const auto kingRing = attacks::kingAttacks(pos.king(c));

            const auto kingRingSquareCount = static_cast<f64>(kingRing.popcount());
            const auto kingRingPieceCount = static_cast<f64>((stmPieces & kingRing).popcount());

            const auto filled = 8.0 * std::min(kingRingPieceCount / kingRingSquareCount, 0.75);

            return kKingRingPieceScale * static_cast<i32>(std::pow(filled, 1.6));
        }
    } // namespace

    Score staticEval(const Position& pos) {
        const auto stm = pos.stm();
        const auto nstm = pos.stm().flip();

        Score score{};

        score += evalMaterial(pos, stm) - evalMaterial(pos, nstm);
        score += evalKingSafety(pos, stm) - evalKingSafety(pos, nstm);

        return std::clamp(score, -kScoreWin + 1, kScoreWin - 1);
    }
} // namespace stoat::eval
