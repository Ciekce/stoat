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

#include "material.h"

namespace stoat::eval {
    Score staticEval(const Position& pos) {
        const auto imbalance = [&](PieceType pt) {
            const auto count = pos.pieceBb(pt, pos.stm()).popcount() - pos.pieceBb(pt, pos.stm().flip()).popcount();
            return count * pieceValue(pt);
        };

        const auto handValue = [](const Hand& hand) {
            const auto handPieceValue = [&](PieceType pt) { return static_cast<i32>(hand.count(pt)) * pieceValue(pt); };

            if (hand.empty()) {
                return 0;
            }

            Score value{};

            value += handPieceValue(PieceTypes::kPawn);
            value += handPieceValue(PieceTypes::kLance);
            value += handPieceValue(PieceTypes::kKnight);
            value += handPieceValue(PieceTypes::kSilver);
            value += handPieceValue(PieceTypes::kGold);
            value += handPieceValue(PieceTypes::kBishop);
            value += handPieceValue(PieceTypes::kRook);

            return value;
        };

        Score score{};

        score += imbalance(PieceTypes::kPawn);
        score += imbalance(PieceTypes::kPromotedPawn);
        score += imbalance(PieceTypes::kLance);
        score += imbalance(PieceTypes::kKnight);
        score += imbalance(PieceTypes::kPromotedLance);
        score += imbalance(PieceTypes::kPromotedKnight);
        score += imbalance(PieceTypes::kSilver);
        score += imbalance(PieceTypes::kPromotedSilver);
        score += imbalance(PieceTypes::kGold);
        score += imbalance(PieceTypes::kBishop);
        score += imbalance(PieceTypes::kRook);
        score += imbalance(PieceTypes::kPromotedBishop);
        score += imbalance(PieceTypes::kPromotedRook);

        score += handValue(pos.hand(pos.stm()));
        score -= handValue(pos.hand(pos.stm().flip()));

        return score;
    }
} // namespace stoat::eval
