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

#include "../attacks/attacks.h"
#include "material.h"

namespace stoat::eval {
    namespace {
        constexpr Score kLanceMobility = 7;
        constexpr Score kKnightMobility = 13;
        constexpr Score kSilverMobility = 16;
        constexpr Score kGoldMobility = 16;
        constexpr Score kBishopMobility = 6;
        constexpr Score kRookMobility = 7;
        constexpr Score kPromotedBishopMobility = 6;
        constexpr Score kPromotedRookMobility = 7;

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

        [[nodiscard]] Score evalMobility(const Position& pos, Color c) {
            const auto ourPieces = pos.colorBb(c);
            const auto occ = pos.occupancy();

            const auto mobilityCount = [&](Bitboard bb, auto attackGetter) {
                i32 count{};

                while (!bb.empty()) {
                    const auto sq = bb.popLsb();
                    const auto attacks = attackGetter(sq, c) & ~ourPieces;
                    count += attacks.popcount();
                }

                return count;
            };

            const auto golds = pos.pieceBb(PieceTypes::kGold, c) | pos.pieceBb(PieceTypes::kPromotedPawn, c)
                             | pos.pieceBb(PieceTypes::kPromotedLance, c) | pos.pieceBb(PieceTypes::kPromotedKnight, c)
                             | pos.pieceBb(PieceTypes::kPromotedSilver, c);

            Score score{};

            score += kLanceMobility * mobilityCount(pos.pieceBb(PieceTypes::kLance, c), [&](Square sq, Color c) {
                         return attacks::lanceAttacks(sq, c, occ);
                     });

            score += kKnightMobility * mobilityCount(pos.pieceBb(PieceTypes::kKnight, c), attacks::knightAttacks);
            score += kSilverMobility * mobilityCount(pos.pieceBb(PieceTypes::kSilver, c), attacks::silverAttacks);
            score += kGoldMobility * mobilityCount(golds, attacks::goldAttacks);

            score += kBishopMobility * mobilityCount(pos.pieceBb(PieceTypes::kBishop, c), [&](Square sq, Color c) {
                         return attacks::bishopAttacks(sq, occ);
                     });

            score += kRookMobility * mobilityCount(pos.pieceBb(PieceTypes::kRook, c), [&](Square sq, Color c) {
                         return attacks::rookAttacks(sq, occ);
                     });

            score += kPromotedBishopMobility
                   * mobilityCount(pos.pieceBb(PieceTypes::kPromotedBishop, c), [&](Square sq, Color c) {
                         return attacks::promotedBishopAttacks(sq, occ);
                     });

            score += kPromotedRookMobility
                   * mobilityCount(pos.pieceBb(PieceTypes::kPromotedRook, c), [&](Square sq, Color c) {
                         return attacks::promotedRookAttacks(sq, occ);
                     });

            return score;
        }
    } // namespace

    Score staticEval(const Position& pos) {
        const auto stm = pos.stm();
        const auto nstm = pos.stm().flip();

        Score score{};

        score += evalMaterial(pos, stm) - evalMaterial(pos, nstm);
        score += evalMobility(pos, stm) - evalMobility(pos, nstm);

        return score;
    }
} // namespace stoat::eval
