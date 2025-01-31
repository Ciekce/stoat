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

#include "../attacks/attacks.h"
#include "material.h"

namespace stoat::eval {
    namespace {
        constexpr std::array kKingRingAttackPower = {
            10, // pawn
            25, // promoted pawn
            12, // lance
            10, // knight
            25, // promoted lance
            25, // promoted knight
            20, // silver
            25, // promoted silver
            27, // gold
            20, // bishop
            25, // rook,
            35, // promoted bishop
            37, // promoted rook
            15, // king
        };

        constexpr std::array kKingRingPromoAreaAttackPower = {
            25, // pawn
            25, // promoted pawn
            25, // lance
            25, // knight
            25, // promoted lance
            25, // promoted knight
            25, // silver
            25, // promoted silver
            27, // gold
            35, // bishop
            37, // rook,
            35, // promoted bishop
            37, // promoted rook
            15, // king
        };

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
            const auto kingRing = attacks::kingAttacks(pos.king(c));
            const auto nstmPromoArea = Bitboards::promoArea(c.flip());

            const auto occ = pos.occupancy();

            Score score{};

            auto oppPieces = pos.colorBb(c.flip());
            while (!oppPieces.empty()) {
                const auto sq = oppPieces.popLsb();
                const auto pt = pos.pieceOn(sq).type();

                const auto kingRingAttacks = kingRing & attacks::pieceAttacks(pt, sq, c, occ);

                score -= kKingRingAttackPower[pt.idx()] * (kingRingAttacks & ~nstmPromoArea).popcount();
                score -= kKingRingPromoAreaAttackPower[pt.idx()] * (kingRingAttacks & nstmPromoArea).popcount();
            }

            return score;
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
