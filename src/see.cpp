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

#include "see.h"

#include <algorithm>

#include "attacks/attacks.h"
#include "rays.h"

namespace stoat::see {
    namespace {
        [[nodiscard]] i32 gain(const Position& pos, Move move) {
            // perhaps unintuitively, dropping a piece does not actually
            // change the material balance, so it does not gain anything
            if (move.isDrop()) {
                return 0;
            }

            const auto captured = pos.pieceOn(move.to());
            auto gain = pieceValue(captured.typeOrNone());

            return gain;
        }

        [[nodiscard]] PieceType popLeastValuable(const Position& pos, Bitboard& occ, Bitboard attackers, Color c) {
            assert(c);

            // sort pieces into ascending order of value, tiebreaking by piece id order
            //TODO stable_sort once it's constexpr
            static constexpr auto kOrderedPieces = [] {
                auto pieces = PieceTypes::kAll;
                std::ranges::sort(pieces, [](PieceType a, PieceType b) {
                    return b == PieceTypes::kKing
                        || (a != PieceTypes::kKing
                            && (pieceValue(a) * 1000 + a.idx()) < (pieceValue(b) * 1000 + b.idx()));
                });
                return pieces;
            }();

            for (const auto pt : kOrderedPieces) {
                const auto ptAttackers = attackers & pos.pieceBb(pt, c);
                if (!ptAttackers.empty()) {
                    occ ^= ptAttackers.isolateLsb();
                    return pt;
                }
            }

            return PieceTypes::kNone;
        }

        [[nodiscard]] constexpr bool canMoveDiagonally(PieceType pt) {
            assert(pt);
            return pt == PieceTypes::kPromotedPawn || pt == PieceTypes::kPromotedLance
                || pt == PieceTypes::kPromotedKnight || pt == PieceTypes::kSilver || pt == PieceTypes::kPromotedSilver
                || pt == PieceTypes::kGold || pt == PieceTypes::kBishop || pt == PieceTypes::kPromotedBishop
                || pt == PieceTypes::kPromotedRook;
        }

        [[nodiscard]] constexpr bool canMoveOrthogonally(PieceType pt) {
            assert(pt);
            return pt == PieceTypes::kPawn || pt == PieceTypes::kPromotedPawn || pt == PieceTypes::kLance
                || pt == PieceTypes::kPromotedLance || pt == PieceTypes::kPromotedKnight || pt == PieceTypes::kSilver
                || pt == PieceTypes::kPromotedSilver || pt == PieceTypes::kGold || pt == PieceTypes::kRook
                || pt == PieceTypes::kPromotedBishop || pt == PieceTypes::kPromotedRook;
        }
    } // namespace

    bool see(const Position& pos, Move move, i32 threshold) {
        const auto stm = pos.stm();

        auto score = gain(pos, move) - threshold;

        if (score < 0) {
            return false;
        }

        auto next = move.isDrop() ? move.dropPiece() : pos.pieceOn(move.from()).type();

        score -= pieceValue(next);

        if (score >= 0) {
            return true;
        }

        const auto sq = move.to();
        auto occ = pos.occupancy() ^ sq.bit();

        if (!move.isDrop()) {
            occ ^= move.from().bit();
        }

        const auto lances = pos.pieceTypeBb(PieceTypes::kLance);
        const auto bishops = pos.pieceTypeBb(PieceTypes::kBishop) | pos.pieceTypeBb(PieceTypes::kPromotedBishop);
        const auto rooks = pos.pieceTypeBb(PieceTypes::kRook) | pos.pieceTypeBb(PieceTypes::kPromotedRook);

        const auto blackPinned = pos.pinned(Colors::kBlack);
        const auto whitePinned = pos.pinned(Colors::kWhite);

        const auto blackKingRay = rayIntersecting(pos.kingSq(Colors::kBlack), sq);
        const auto whiteKingRay = rayIntersecting(pos.kingSq(Colors::kWhite), sq);

        const auto allowed = ~(blackPinned | whitePinned) | (blackPinned & blackKingRay) | (whitePinned & whiteKingRay);

        auto attackers = pos.allAttackersTo(sq, occ) & allowed;

        auto curr = stm.flip();

        while (true) {
            const auto currAttackers = attackers & pos.colorBb(curr);

            if (currAttackers.empty()) {
                break;
            }

            next = popLeastValuable(pos, occ, attackers, curr);

            if (canMoveDiagonally(next)) {
                attackers |= attacks::bishopAttacks(sq, occ) & bishops;
            }

            if (canMoveOrthogonally(next)) {
                const auto rookAttacks = attacks::rookAttacks(sq, occ);

                attackers |= rookAttacks & Bitboards::kFiles[sq.file()] & lances;
                attackers |= rookAttacks & rooks;
            }

            attackers &= occ;

            score = -score - 1 - pieceValue(next);
            curr = curr.flip();

            if (score >= 0) {
                if (next == PieceTypes::kKing && !(attackers & pos.colorBb(curr)).empty()) {
                    curr = curr.flip();
                }
                break;
            }
        }

        return curr != stm;
    }
} // namespace stoat::see
