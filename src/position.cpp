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

#include "position.h"

#include <algorithm>
#include <iterator>

#include "attacks/attacks.h"
#include "eval/nnue.h"
#include "keys.h"
#include "movegen.h"
#include "rays.h"
#include "util/parse.h"
#include "util/split.h"

namespace stoat {
    namespace {
        constexpr i32 kPawnHandBits = 5;
        constexpr i32 kLanceHandBits = 3;
        constexpr i32 kKnightHandBits = 3;
        constexpr i32 kSilverHandBits = 3;
        constexpr i32 kGoldHandBits = 3;
        constexpr i32 kBishopHandBits = 2;
        constexpr i32 kRookHandBits = 2;

        constexpr i32 kPawnHandOffset = 0;
        constexpr i32 kLanceHandOffset = kPawnHandOffset + kPawnHandBits;
        constexpr i32 kKnightHandOffset = kLanceHandOffset + kLanceHandBits;
        constexpr i32 kSilverHandOffset = kKnightHandOffset + kKnightHandBits;
        constexpr i32 kGoldHandOffset = kSilverHandOffset + kSilverHandBits;
        constexpr i32 kBishopHandOffset = kGoldHandOffset + kGoldHandBits;
        constexpr i32 kRookHandOffset = kBishopHandOffset + kBishopHandBits;

        static_assert(kRookHandOffset + kRookHandBits <= 32);

        constexpr auto kHandOffsets = [] {
            std::array<i32, PieceTypes::kCount> offsets{};
            offsets.fill(-1);

            offsets[PieceTypes::kPawn.idx()] = kPawnHandOffset;
            offsets[PieceTypes::kLance.idx()] = kLanceHandOffset;
            offsets[PieceTypes::kKnight.idx()] = kKnightHandOffset;
            offsets[PieceTypes::kSilver.idx()] = kSilverHandOffset;
            offsets[PieceTypes::kGold.idx()] = kGoldHandOffset;
            offsets[PieceTypes::kBishop.idx()] = kBishopHandOffset;
            offsets[PieceTypes::kRook.idx()] = kRookHandOffset;

            return offsets;
        }();

        constexpr auto kHandMasks = [] {
            std::array<u32, PieceTypes::kCount> masks{};

            masks[PieceTypes::kPawn.idx()] = ((1 << kPawnHandBits) - 1) << kPawnHandOffset;
            masks[PieceTypes::kLance.idx()] = ((1 << kLanceHandBits) - 1) << kLanceHandOffset;
            masks[PieceTypes::kKnight.idx()] = ((1 << kKnightHandBits) - 1) << kKnightHandOffset;
            masks[PieceTypes::kSilver.idx()] = ((1 << kSilverHandBits) - 1) << kSilverHandOffset;
            masks[PieceTypes::kGold.idx()] = ((1 << kGoldHandBits) - 1) << kGoldHandOffset;
            masks[PieceTypes::kBishop.idx()] = ((1 << kBishopHandBits) - 1) << kBishopHandOffset;
            masks[PieceTypes::kRook.idx()] = ((1 << kRookHandBits) - 1) << kRookHandOffset;

            return masks;
        }();
    } // namespace

    u32 Hand::count(PieceType pt) const {
        assert(pt);

        const auto offset = kHandOffsets[pt.idx()];
        const auto mask = kHandMasks[pt.idx()];

        assert(offset != -1);

        return (m_hand & mask) >> offset;
    }

    u32 Hand::increment(PieceType pt) {
        assert(pt);
        const auto curr = count(pt);
        assert(curr < (kHandMasks[pt.idx()] >> kHandOffsets[pt.idx()]));
        set(pt, curr + 1);
        return curr + 1;
    }

    u32 Hand::decrement(PieceType pt) {
        assert(pt);
        const auto curr = count(pt);
        assert(curr > 0);
        set(pt, curr - 1);
        return curr - 1;
    }

    void Hand::set(PieceType pt, u32 count) {
        assert(pt);

        const auto offset = kHandOffsets[pt.idx()];
        const auto mask = kHandMasks[pt.idx()];

        assert(offset != -1);
        assert(count <= (mask >> offset));

        m_hand = (m_hand & ~mask) | (count << offset);
    }

    Hand Hand::fromRaw(u32 raw) {
        return Hand{raw};
    }

    std::string Hand::sfen(bool uppercase) const {
        std::string sfen{};
        auto itr = std::back_inserter(sfen);

        const auto print = [&](PieceType pt) {
            const auto count = this->count(pt);

            if (count == 0) {
                return;
            }

            if (count > 1) {
                fmt::format_to(itr, "{}", count);
            }

            const auto c = pt.str()[0];
            assert(c != '+' && c != '?');

            fmt::format_to(itr, "{}", uppercase ? c : static_cast<char>(std::tolower(c)));
        };

        print(PieceTypes::kRook);
        print(PieceTypes::kBishop);
        print(PieceTypes::kGold);
        print(PieceTypes::kSilver);
        print(PieceTypes::kKnight);
        print(PieceTypes::kLance);
        print(PieceTypes::kPawn);

        return sfen;
    }

    void PositionKeys::clear() {
        all = 0;
        castle = 0;
        cavalry = 0;
        hand = 0;
    }

    void PositionKeys::flipPiece(Piece piece, Square sq) {
        assert(piece);
        assert(sq);

        const auto key = keys::pieceSquare(piece, sq);

        all ^= key;

        if (piece.type() == PieceTypes::kKing || piece.type() == PieceTypes::kSilver
            || piece.type() == PieceTypes::kGold)
        {
            castle ^= key;
        }
        if (piece.type() == PieceTypes::kKnight || piece.type() == PieceTypes::kBishop
            || piece.type() == PieceTypes::kRook || piece.type() == PieceTypes::kPromotedBishop
            || piece.type() == PieceTypes::kPromotedRook)
        {
            cavalry ^= key;
        }
        if (piece.type() == PieceTypes::kKing) {
            hand ^= key;
        }
    }

    void PositionKeys::movePiece(Piece piece, Square from, Square to) {
        assert(piece);
        assert(from);
        assert(to);

        const auto key = keys::pieceSquare(piece, from) ^ keys::pieceSquare(piece, to);

        all ^= key;

        if (piece.type() == PieceTypes::kKing || piece.type() == PieceTypes::kSilver
            || piece.type() == PieceTypes::kGold)
        {
            castle ^= key;
        }
        if (piece.type() == PieceTypes::kKnight || piece.type() == PieceTypes::kBishop
            || piece.type() == PieceTypes::kRook || piece.type() == PieceTypes::kPromotedBishop
            || piece.type() == PieceTypes::kPromotedRook)
        {
            cavalry ^= key;
        }
        if (piece.type() == PieceTypes::kKing) {
            hand ^= key;
        }
    }

    void PositionKeys::flipStm() {
        all ^= keys::stm();
    }

    void PositionKeys::flipHandCount(Color c, PieceType pt, u32 count) {
        assert(c);
        assert(pt);
        assert(count <= maxPiecesInHand(pt));

        const auto key = keys::pieceInHand(c, pt, count);

        all ^= key;
        hand ^= key;

        if (pt == PieceTypes::kKnight || pt == PieceTypes::kBishop || pt == PieceTypes::kRook) {
            cavalry ^= key;
        }
    }

    void PositionKeys::switchHandCount(Color c, PieceType pt, u32 before, u32 after) {
        assert(c);
        assert(pt);
        assert(before <= maxPiecesInHand(pt));
        assert(after <= maxPiecesInHand(pt));

        const auto key = keys::pieceInHand(c, pt, before) ^ keys::pieceInHand(c, pt, after);

        all ^= key;
        hand ^= key;

        if (pt == PieceTypes::kKnight || pt == PieceTypes::kBishop || pt == PieceTypes::kRook) {
            cavalry ^= key;
        }
    }

    template Position Position::applyMove<NnueUpdateAction::kNone>(Move, eval::nnue::NnueState*) const;
    template Position Position::applyMove<NnueUpdateAction::kPush>(Move, eval::nnue::NnueState*) const;
    template Position Position::applyMove<NnueUpdateAction::kApplyInPlace>(Move, eval::nnue::NnueState*) const;

    Position::Position() {
        m_mailbox.fill(Pieces::kNone);
    }

    template <NnueUpdateAction kUpdateAction>
    Position Position::applyMove(Move move, eval::nnue::NnueState* nnueState) const {
        static constexpr bool kUpdateNnue = kUpdateAction != NnueUpdateAction::kNone;

        auto newPos = *this;
        eval::nnue::NnueUpdates updates{};

        const auto stm = this->stm();

        if (move.isDrop()) {
            const auto sq = move.to();
            const auto piece = move.dropPiece().withColor(stm);

            newPos.dropPiece<kUpdateNnue>(sq, piece, updates);
        } else {
            const auto to = move.to();
            const auto from = move.from();

            const auto piece = newPos.pieceOn(from);

            if (move.isPromo()) {
                newPos.promotePiece<kUpdateNnue>(from, to, piece, updates);
            } else {
                newPos.movePiece<kUpdateNnue>(from, to, piece, updates);
            }
        }

        if constexpr (kUpdateAction == NnueUpdateAction::kPush) {
            nnueState->push(newPos, updates);
        } else if constexpr (kUpdateAction == NnueUpdateAction::kApplyInPlace) {
            nnueState->applyInPlace(newPos, updates);
        }

        ++newPos.m_moveCount;

        newPos.m_stm = newPos.m_stm.flip();
        newPos.m_keys.flipStm();

        newPos.updateAttacks();

        if (newPos.isInCheck()) {
            ++newPos.m_consecutiveChecks[newPos.stm().idx()];
        } else {
            newPos.m_consecutiveChecks[newPos.stm().idx()] = 0;
        }

        return newPos;
    }

    Position Position::applyNullMove() const {
        auto newPos = *this;

        ++newPos.m_moveCount;

        newPos.m_stm = newPos.m_stm.flip();
        newPos.m_keys.flipStm();

        newPos.updateAttacks();

        return newPos;
    }

    u64 Position::keyAfter(Move move) const {
        auto key = m_keys.all ^ keys::stm();

        const auto& hand = this->hand(m_stm);

        if (move.isDrop()) {
            key ^= keys::pieceSquare(move.dropPiece().withColor(m_stm), move.to());

            const auto count = hand.count(move.dropPiece());
            key ^= keys::pieceInHand(m_stm, move.dropPiece(), count);
            key ^= keys::pieceInHand(m_stm, move.dropPiece(), count - 1);

            return key;
        }

        const auto moving = pieceOn(move.from());
        const auto captured = pieceOn(move.to());

        key ^= keys::pieceSquare(moving, move.from());

        if (move.isPromo()) {
            key ^= keys::pieceSquare(moving.promoted(), move.to());
        } else {
            key ^= keys::pieceSquare(moving, move.to());
        }

        if (captured) {
            key ^= keys::pieceSquare(captured, move.to());

            const auto handPt = captured.type().unpromoted();
            const auto count = hand.count(handPt);
            key ^= keys::pieceInHand(m_stm, handPt, count);
            key ^= keys::pieceInHand(m_stm, handPt, count + 1);
        }

        return key;
    }

    SennichiteStatus Position::testSennichite(
        bool cuteChessWorkaround,
        std::span<const u64> keyHistory,
        i32 limit
    ) const {
        const auto end = std::max(0, static_cast<i32>(keyHistory.size()) - limit - 1);

        i32 repetitions = 1;

        for (i32 i = static_cast<i32>(keyHistory.size()) - 4; i >= end; i -= 2) {
            if (keyHistory[i] == key()) {
                --repetitions;
                if (repetitions == 0) {
                    // Older cutechess versions do not handle perpetuals
                    // properly - work around that to avoid illegal moves
                    if (cuteChessWorkaround) {
                        return isInCheck() ? SennichiteStatus::kWin : SennichiteStatus::kDraw;
                    } else {
                        return m_consecutiveChecks[stm().idx()] >= 2 ? SennichiteStatus::kWin : SennichiteStatus::kDraw;
                    }
                }
            }
        }

        return SennichiteStatus::kNone;
    }

    bool Position::isEnteringKingsWin() const {
        if (isInCheck()) {
            return false;
        }

        const auto stm = this->stm();
        const auto promoZone = Bitboards::promoArea(stm);

        const auto king = pieceBb(PieceTypes::kKing, stm);

        if ((promoZone & king).empty()) {
            return false;
        }

        if ((promoZone & colorBb(stm)).popcount() < 11) { // 10 pieces other than the king, plus the king
            return false;
        }

        const auto bishopsRooks = pieceBb(PieceTypes::kBishop, stm) | pieceBb(PieceTypes::kRook, stm)
                                | pieceBb(PieceTypes::kPromotedBishop, stm) | pieceBb(PieceTypes::kPromotedRook, stm);

        const auto hand = this->hand(stm);

        i32 score = 0;

        score += 5 * (promoZone & bishopsRooks).popcount();
        score += (promoZone & colorBb(stm) ^ king ^ bishopsRooks).popcount();
        score += hand.count(PieceTypes::kPawn) + hand.count(PieceTypes::kLance) + hand.count(PieceTypes::kKnight)
               + hand.count(PieceTypes::kSilver) + hand.count(PieceTypes::kGold)
               + 5 * (hand.count(PieceTypes::kBishop) + hand.count(PieceTypes::kRook));
        score += stm == Colors::kWhite;

        return score >= 28;
    }

    bool Position::isPseudolegal(Move move) const {
        assert(!move.isNull());

        const auto stm = this->stm();

        const auto occ = occupancy();

        const auto getPromoRequiredZone = [&](PieceType pt) {
            Bitboard zone{};

            // pawns, knights and lances must promote upon reaching the backrank
            if (pt == PieceTypes::kPawn || pt == PieceTypes::kLance || pt == PieceTypes::kKnight) {
                zone |= Bitboards::relativeRank(stm, 8);
            }

            // knights additionally must promote upon reaching the rank before the backrank
            if (pt == PieceTypes::kKnight) {
                zone |= Bitboards::relativeRank(stm, 7);
            }

            return zone;
        };

        if (move.isDrop()) {
            // can't drop a piece we don't have
            const auto& hand = this->hand(stm);
            if (hand.count(move.dropPiece()) == 0) {
                return false;
            }

            // must drop in an empty square
            if (occ.getSquare(move.to())) {
                return false;
            }

            // can't drop a piece on a square where it would
            // be required to promote when moving (see above)
            const auto promoRequiredZone = getPromoRequiredZone(move.dropPiece());
            if (promoRequiredZone.getSquare(move.to())) {
                return false;
            }

            // can't drop a pawn in the same file as another of our unpromoted pawns
            if (move.dropPiece() == PieceTypes::kPawn) {
                const auto stmPawnFiles = pieceBb(PieceTypes::kPawn, stm).fillFile();
                if (stmPawnFiles.getSquare(move.to())) {
                    return false;
                }
            }

            return true;
        }

        const auto moving = pieceOn(move.from());

        // must move a piece that a) exists and b) is our own
        if (!moving || moving.color() != stm) {
            return false;
        }

        const auto captured = pieceOn(move.to());

        // can't capture our own piece, or a king
        if (captured && (captured.color() == stm || captured.type() == PieceTypes::kKing)) {
            return false;
        }

        if (move.isPromo()) {
            // can't promote a gold, a king, or an already-promoted piece
            if (!moving.type().canPromote()) {
                return false;
            }

            // can only promote when moving into, in, or out of our promo area
            const auto promoArea = Bitboards::promoArea(stm);
            if (!promoArea.getSquare(move.from()) && !promoArea.getSquare(move.to())) {
                return false;
            }
        } else {
            const auto promoRequiredZone = getPromoRequiredZone(moving.type());
            if (promoRequiredZone.getSquare(move.to())) {
                return false;
            }
        }

        // have to actually move to a square that this piece can attack
        const auto attacks = attacks::pieceAttacks(moving.type(), move.from(), stm, occ);
        if (!attacks.getSquare(move.to())) {
            return false;
        }

        return true;
    }

    // Assumes moves are pseudolegal
    bool Position::isLegal(Move move) const {
        assert(!move.isNull());

        const auto stm = this->stm();
        const auto nstm = this->stm().flip();

        const auto stmKing = kingSq(stm);

        if (move.isDrop()) {
            if (isInCheck()) {
                // multiple checks can only be evaded with a king move
                if (m_checkers.multiple()) {
                    return false;
                }

                const auto checker = m_checkers.lsb();
                const auto checkRay = rayBetween(stmKing, checker);

                // a drop must block the check
                if (!checkRay.getSquare(move.to())) {
                    return false;
                }
            }

            // pawn drop mate rule (delivering mate by dropping a pawn is illegal)
            if (move.dropPiece() == PieceTypes::kPawn) {
                const auto dropBb = Bitboard::fromSquare(move.to());
                if (!(dropBb.shiftNorthRelative(stm) & pieceBb(PieceTypes::kKing, nstm)).empty()) {
                    // this pawn drop gives check - ensure it's not mate
                    // slow and cursed, but rare
                    const auto newPos = applyMove(move);

                    assert(newPos.checkers().one());
                    assert(newPos.checkers().getSquare(move.to()));

                    movegen::MoveList newMoves{};
                    movegen::generateAll(newMoves, newPos);

                    return std::ranges::any_of(newMoves, [&](Move newMove) { return newPos.isLegal(newMove); });
                }
            }

            // impossible to put yourself in check by dropping a piece, no cannons here
            return true;
        }

        if (pieceOn(move.from()).type() == PieceTypes::kKing) {
            // remove the king to account for moving away from the checker
            const auto kinglessOcc = occupancy() ^ pieceBb(PieceTypes::kKing, stm);
            return !isAttacked(move.to(), nstm, kinglessOcc);
        } else if (m_checkers.multiple()) {
            // multiple checks can only be evaded with a king move
            return false;
        }

        if (m_pinned.getSquare(move.from())) {
            const auto pinRay = rayIntersecting(move.from(), stmKing);
            if (!pinRay.getSquare(move.to())) {
                return false;
            }
        }

        if (isInCheck()) {
            const auto checker = m_checkers.lsb();
            // includes the checker
            const auto checkRay = rayBetween(stmKing, checker) | checker.bit();

            // must either block the check or capture the checker
            if (!checkRay.getSquare(move.to())) {
                return false;
            }
        }

        // :3
        return true;
    }

    bool Position::isCapture(Move move) const {
        return pieceOn(move.to()) != Pieces::kNone;
    }

    bool Position::isAttacked(Square sq, Color attacker, Bitboard occ) const {
        assert(sq);
        assert(attacker);

        const auto c = attacker.flip();

        const auto horses = pieceBb(PieceTypes::kPromotedBishop, attacker);
        const auto dragons = pieceBb(PieceTypes::kPromotedRook, attacker);

        const auto rooks = dragons | pieceBb(PieceTypes::kRook, attacker);

        if (const auto pawns = pieceBb(PieceTypes::kPawn, attacker); !(pawns & attacks::pawnAttacks(sq, c)).empty()) {
            return true;
        }

        if (const auto knights = pieceBb(PieceTypes::kKnight, attacker);
            !(knights & attacks::knightAttacks(sq, c)).empty())
        {
            return true;
        }

        if (const auto silvers = pieceBb(PieceTypes::kSilver, attacker);
            !(silvers & attacks::silverAttacks(sq, c)).empty())
        {
            return true;
        }

        if (const auto golds = pieceBb(PieceTypes::kGold, attacker) | pieceBb(PieceTypes::kPromotedPawn, attacker)
                             | pieceBb(PieceTypes::kPromotedLance, attacker)
                             | pieceBb(PieceTypes::kPromotedKnight, attacker)
                             | pieceBb(PieceTypes::kPromotedSilver, attacker);
            !(golds & attacks::goldAttacks(sq, c)).empty())
        {
            return true;
        }

        if (const auto kings = horses | dragons | pieceBb(PieceTypes::kKing, attacker);
            !(kings & attacks::kingAttacks(sq)).empty())
        {
            return true;
        }

        if (const auto lances = rooks | pieceBb(PieceTypes::kLance, attacker);
            !(lances & attacks::lanceAttacks(sq, c, occ)).empty())
        {
            return true;
        }

        if (const auto bishops = horses | pieceBb(PieceTypes::kBishop, attacker);
            !(bishops & attacks::bishopAttacks(sq, occ)).empty())
        {
            return true;
        }

        if (!(rooks & attacks::rookAttacks(sq, occ)).empty()) {
            return true;
        }

        return false;
    }

    Bitboard Position::attackersTo(Square sq, Color attacker) const {
        assert(sq);
        assert(attacker);

        const auto defender = attacker.flip();
        const auto occ = occupancy();

        Bitboard attackers{};

        const auto horses = pieceBb(PieceTypes::kPromotedBishop, attacker);
        const auto dragons = pieceBb(PieceTypes::kPromotedRook, attacker);

        const auto pawns = pieceBb(PieceTypes::kPawn, attacker);
        attackers |= pawns & attacks::pawnAttacks(sq, defender);

        const auto lances = pieceBb(PieceTypes::kLance, attacker);
        attackers |= lances & attacks::lanceAttacks(sq, defender, occ);

        const auto knights = pieceBb(PieceTypes::kKnight, attacker);
        attackers |= knights & attacks::knightAttacks(sq, defender);

        const auto silvers = pieceBb(PieceTypes::kSilver, attacker);
        attackers |= silvers & attacks::silverAttacks(sq, defender);

        const auto golds = pieceBb(PieceTypes::kGold, attacker) | pieceBb(PieceTypes::kPromotedPawn, attacker)
                         | pieceBb(PieceTypes::kPromotedLance, attacker)
                         | pieceBb(PieceTypes::kPromotedKnight, attacker)
                         | pieceBb(PieceTypes::kPromotedSilver, attacker);
        attackers |= golds & attacks::goldAttacks(sq, defender);

        const auto bishops = horses | pieceBb(PieceTypes::kBishop, attacker);
        attackers |= bishops & attacks::bishopAttacks(sq, occ);

        const auto rooks = dragons | pieceBb(PieceTypes::kRook, attacker);
        attackers |= rooks & attacks::rookAttacks(sq, occ);

        const auto kings = horses | dragons | pieceBb(PieceTypes::kKing, attacker);
        attackers |= kings & attacks::kingAttacks(sq);

        return attackers;
    }

    Bitboard Position::allAttackersTo(Square sq, Bitboard occ) const {
        assert(sq);

        Bitboard attackers{};

        const auto black = colorBb(Colors::kBlack);
        const auto white = colorBb(Colors::kWhite);

        const auto horses = pieceTypeBb(PieceTypes::kPromotedBishop);
        const auto dragons = pieceTypeBb(PieceTypes::kPromotedRook);

        const auto pawns = pieceTypeBb(PieceTypes::kPawn);

        attackers |= pawns & black & attacks::pawnAttacks(sq, Colors::kWhite);
        attackers |= pawns & white & attacks::pawnAttacks(sq, Colors::kBlack);

        const auto lances = pieceTypeBb(PieceTypes::kLance);

        attackers |= lances & black & attacks::lanceAttacks(sq, Colors::kWhite, occ);
        attackers |= lances & white & attacks::lanceAttacks(sq, Colors::kBlack, occ);

        const auto knights = pieceTypeBb(PieceTypes::kKnight);

        attackers |= knights & black & attacks::knightAttacks(sq, Colors::kWhite);
        attackers |= knights & white & attacks::knightAttacks(sq, Colors::kBlack);

        const auto silvers = pieceTypeBb(PieceTypes::kSilver);

        attackers |= silvers & black & attacks::silverAttacks(sq, Colors::kWhite);
        attackers |= silvers & white & attacks::silverAttacks(sq, Colors::kBlack);

        const auto golds = pieceTypeBb(PieceTypes::kGold) | pieceTypeBb(PieceTypes::kPromotedPawn)
                         | pieceTypeBb(PieceTypes::kPromotedLance) | pieceTypeBb(PieceTypes::kPromotedKnight)
                         | pieceTypeBb(PieceTypes::kPromotedSilver);

        attackers |= golds & black & attacks::goldAttacks(sq, Colors::kWhite);
        attackers |= golds & white & attacks::goldAttacks(sq, Colors::kBlack);

        const auto bishops = horses | pieceTypeBb(PieceTypes::kBishop);
        attackers |= bishops & attacks::bishopAttacks(sq, occ);

        const auto rooks = dragons | pieceTypeBb(PieceTypes::kRook);
        attackers |= rooks & attacks::rookAttacks(sq, occ);

        const auto kings = horses | dragons | pieceTypeBb(PieceTypes::kKing);
        attackers |= kings & attacks::kingAttacks(sq);

        return attackers;
    }

    std::string Position::sfen() const {
        std::string sfen{};
        auto itr = std::back_inserter(sfen);

        for (i32 rank = 8; rank >= 0; --rank) {
            for (i32 file = 0; file < 9; ++file) {
                if (const auto piece = pieceOn(Square::fromFileRank(file, rank)); piece == Pieces::kNone) {
                    u32 emptySquares = 1;
                    for (; file < 8 && pieceOn(Square::fromFileRank(file + 1, rank)) == Pieces::kNone;
                         ++file, ++emptySquares)
                    {}

                    fmt::format_to(itr, "{}", emptySquares);
                } else {
                    fmt::format_to(itr, "{}", piece);
                }
            }

            if (rank > 0) {
                fmt::format_to(itr, "/");
            }
        }

        fmt::format_to(itr, "{}", stm() == Colors::kBlack ? " b " : " w ");

        const auto& blackHand = hand(Colors::kBlack);
        const auto& whiteHand = hand(Colors::kWhite);

        if (blackHand.empty() && whiteHand.empty()) {
            fmt::format_to(itr, "-");
        } else {
            fmt::format_to(itr, "{}{}", blackHand.sfen(true), whiteHand.sfen(false));
        }

        fmt::format_to(itr, " {}", moveCount());

        return sfen;
    }

    void Position::regenKey() {
        m_keys.clear();

        auto occ = occupancy();
        while (!occ.empty()) {
            const auto sq = occ.popLsb();
            const auto piece = pieceOn(sq);

            assert(piece);

            m_keys.flipPiece(piece, sq);
        }

        if (stm() == Colors::kWhite) {
            m_keys.flipStm();
        }

        for (const auto c : {Colors::kBlack, Colors::kWhite}) {
            const auto& hand = this->hand(c);
            for (const auto pt : kHandPieces) {
                m_keys.flipHandCount(c, pt, hand.count(pt));
            }
        }
    }

    void Position::addPiece(Square sq, Piece piece) {
        assert(sq);
        assert(piece);

        assert(!pieceOn(sq));

        m_colors[piece.color().idx()] |= sq.bit();
        m_pieces[piece.type().idx()] |= sq.bit();

        m_mailbox[sq.idx()] = piece;

        m_keys.flipPiece(piece, sq);

        if (piece.type() == PieceTypes::kKing) {
            m_kingSquares.squares[piece.color().idx()] = sq;
        }
    }

    template <bool kUpdateNnue>
    void Position::movePiece(Square from, Square to, Piece piece, eval::nnue::NnueUpdates& nnueUpdates) {
        assert(from);
        assert(to);
        assert(from != to);

        assert(piece);

        const auto captured = pieceOn(to);
        assert(!captured || captured.color() != piece.color());
        assert(!captured || captured.type() != PieceTypes::kKing);

        if (captured) {
            m_colors[captured.color().idx()] ^= to.bit();
            m_pieces[captured.type().idx()] ^= to.bit();

            const auto handPt = captured.type().unpromoted();

            const auto newCount = m_hands[captured.color().flip().idx()].increment(handPt);
            m_keys.switchHandCount(piece.color(), handPt, newCount - 1, newCount);

            m_keys.flipPiece(captured, to);

            if constexpr (kUpdateNnue) {
                nnueUpdates.pushCapture(m_kingSquares, to, captured, newCount - 1);
            }
        }

        m_colors[piece.color().idx()] ^= from.bit() ^ to.bit();
        m_pieces[piece.type().idx()] ^= from.bit() ^ to.bit();

        m_mailbox[from.idx()] = Pieces::kNone;
        m_mailbox[to.idx()] = piece;

        m_keys.movePiece(piece, from, to);

        if (piece.type() == PieceTypes::kKing) {
            m_kingSquares.squares[piece.color().idx()] = to;
            if (kUpdateNnue && eval::nnue::requiresRefresh(piece.color(), to, from)) {
                nnueUpdates.setRefresh(piece.color());
            }
        }

        if constexpr (kUpdateNnue) {
            nnueUpdates.pushMove(m_kingSquares, piece, piece, from, to);
        }
    }

    template <bool kUpdateNnue>
    void Position::promotePiece(Square from, Square to, Piece piece, eval::nnue::NnueUpdates& nnueUpdates) {
        assert(from);
        assert(to);
        assert(from != to);

        assert(piece);
        assert(!piece.isPromoted());

        const auto captured = pieceOn(to);
        assert(!captured || captured.color() != piece.color());
        assert(!captured || captured.type() != PieceTypes::kKing);

        if (captured) {
            m_colors[captured.color().idx()] ^= to.bit();
            m_pieces[captured.type().idx()] ^= to.bit();

            const auto handPt = captured.type().unpromoted();

            const auto newCount = m_hands[captured.color().flip().idx()].increment(handPt);
            m_keys.switchHandCount(piece.color(), handPt, newCount - 1, newCount);

            m_keys.flipPiece(captured, to);

            if constexpr (kUpdateNnue) {
                nnueUpdates.pushCapture(m_kingSquares, to, captured, newCount - 1);
            }
        }

        const auto promoted = piece.promoted();

        m_colors[piece.color().idx()] ^= from.bit() ^ to.bit();

        m_pieces[piece.type().idx()] ^= from.bit();
        m_pieces[promoted.type().idx()] ^= to.bit();

        m_mailbox[from.idx()] = Pieces::kNone;
        m_mailbox[to.idx()] = promoted;

        m_keys.flipPiece(piece, from);
        m_keys.flipPiece(promoted, to);

        // kings cannot promote

        if constexpr (kUpdateNnue) {
            nnueUpdates.pushMove(m_kingSquares, piece, promoted, from, to);
        }
    }

    template <bool kUpdateNnue>
    void Position::dropPiece(Square sq, Piece piece, eval::nnue::NnueUpdates& nnueUpdates) {
        auto& hand = m_hands[piece.color().idx()];

        assert(!pieceOn(sq));
        assert(hand.count(piece.type()) > 0);

        addPiece(sq, piece);

        const auto newCount = hand.decrement(piece.type());
        m_keys.switchHandCount(piece.color(), piece.type(), newCount + 1, newCount);

        if constexpr (kUpdateNnue) {
            nnueUpdates.pushDrop(m_kingSquares, piece, sq, newCount + 1);
        }
    }

    void Position::updateAttacks() {
        const auto stm = this->stm();
        const auto nstm = this->stm().flip();

        m_checkers = attackersTo(kingSq(stm), nstm);
        m_pinned = Bitboards::kEmpty;

        const auto stmKing = kingSq(stm);

        const auto stmOcc = colorBb(stm);
        const auto nstmOcc = colorBb(nstm);

        const auto nstmLances = pieceBb(PieceTypes::kLance, nstm);
        const auto nstmBishops = pieceBb(PieceTypes::kBishop, nstm) | pieceBb(PieceTypes::kPromotedBishop, nstm);
        const auto nstmRooks = pieceBb(PieceTypes::kRook, nstm) | pieceBb(PieceTypes::kPromotedRook, nstm);

        auto potentialAttackers = (attacks::lanceAttacks(stmKing, stm, nstmOcc) & nstmLances)
                                | (attacks::bishopAttacks(stmKing, nstmOcc) & nstmBishops)
                                | (attacks::rookAttacks(stmKing, nstmOcc) & nstmRooks);
        while (!potentialAttackers.empty()) {
            const auto potentialAttacker = potentialAttackers.popLsb();
            const auto maybePinned = stmOcc & rayBetween(potentialAttacker, stmKing);

            if (maybePinned.one()) {
                m_pinned |= maybePinned;
            }
        }
    }

    void Position::regen() {
        m_mailbox.fill(Pieces::kNone);

        for (u8 pieceIdx = 0; pieceIdx < Pieces::kCount; ++pieceIdx) {
            const auto piece = Piece::fromRaw(pieceIdx);

            auto bb = pieceBb(piece);
            while (!bb.empty()) {
                const auto sq = bb.popLsb();
                assert(!m_mailbox[sq.idx()]);
                m_mailbox[sq.idx()] = piece;
            }
        }

        m_kingSquares.squares[Colors::kBlack.idx()] = pieceBb(PieceTypes::kKing, Colors::kBlack).lsb();
        m_kingSquares.squares[Colors::kWhite.idx()] = pieceBb(PieceTypes::kKing, Colors::kWhite).lsb();

        regenKey();
        updateAttacks();
    }

    Position Position::startpos() {
        Position pos{};

        pos.m_colors[Colors::kBlack.idx()] = Bitboard{U128(0, 0x7fd05ff)};
        pos.m_colors[Colors::kWhite.idx()] = Bitboard{U128(0x1ff41, 0x7fc0000000000000)};

        pos.m_pieces[PieceTypes::kPawn.idx()] = Bitboard{U128(0, 0x7fc0000007fc0000)};
        pos.m_pieces[PieceTypes::kLance.idx()] = Bitboard{U128(0x10100, 0x101)};
        pos.m_pieces[PieceTypes::kKnight.idx()] = Bitboard{U128(0x8200, 0x82)};
        pos.m_pieces[PieceTypes::kSilver.idx()] = Bitboard{U128(0x4400, 0x44)};
        pos.m_pieces[PieceTypes::kGold.idx()] = Bitboard{U128(0x2800, 0x28)};
        pos.m_pieces[PieceTypes::kKing.idx()] = Bitboard{U128(0x1000, 0x10)};
        pos.m_pieces[PieceTypes::kBishop.idx()] = Bitboard{U128(0x40, 0x400)};
        pos.m_pieces[PieceTypes::kRook.idx()] = Bitboard{U128(0x1, 0x10000)};

        pos.regen();

        return pos;
    }

    util::Result<Position, SfenError> Position::fromSfenParts(std::span<std::string_view> sfen) {
        if (sfen.size() < 3 || sfen.size() > 4) {
            return util::err<SfenError>("wrong number of SFEN parts");
        }

        Position pos{};

        std::vector<std::string_view> ranks{};
        ranks.reserve(9);

        util::split(ranks, sfen[0], '/');

        if (ranks.size() != 9) {
            return util::err<SfenError>("wrong number of ranks in SFEN");
        }

        for (i32 rankIdx = 0; rankIdx < ranks.size(); ++rankIdx) {
            const auto rank = ranks[rankIdx];

            i32 fileIdx = 0;

            for (i32 curr = 0; curr < rank.size(); ++curr) {
                const auto c = rank[curr];
                if (const auto emptySquares = util::tryParseDigit<i32>(c)) {
                    fileIdx += *emptySquares;
                } else if (c == '+') {
                    if (curr == rank.size() - 1) {
                        return util::err<SfenError>("+ found at end of rank with no matching piece");
                    }

                    const auto pieceStr = rank.substr(curr, 2);

                    if (const auto piece = Piece::fromStr(pieceStr)) {
                        pos.addPiece(Square::fromFileRank(fileIdx, 8 - rankIdx), piece);
                        ++fileIdx;
                        ++curr;
                    } else {
                        return util::err<SfenError>(fmt::format("invalid promoted piece {}", pieceStr));
                    }
                } else if (const auto piece = Piece::fromStr(rank.substr(curr, 1))) {
                    pos.addPiece(Square::fromFileRank(fileIdx, 8 - rankIdx), piece);
                    ++fileIdx;
                } else {
                    return util::err<SfenError>(fmt::format("invalid piece char {}", c));
                }
            }

            if (fileIdx != 9) {
                return util::err<SfenError>("wrong number of files in rank");
            }
        }

        if (const auto blackKingCount = pos.pieceBb(Pieces::kBlackKing).popcount(); blackKingCount != 1) {
            return util::err<SfenError>("black must have exactly 1 king");
        }

        if (const auto whiteKingCount = pos.pieceBb(Pieces::kWhiteKing).popcount(); whiteKingCount != 1) {
            return util::err<SfenError>("white must have exactly 1 king");
        }

        const auto stm = sfen[1];

        if (stm.size() != 1 || (stm[0] != 'b' && stm[0] != 'w')) {
            return util::err<SfenError>("invalid side to move");
        }

        pos.m_stm = stm[0] == 'b' ? Colors::kBlack : Colors::kWhite;

        //TODO ensure opponent is not in check

        const auto hand = sfen[2];

        if (hand != "-") {
            u32 nextCount = 1;

            for (i32 curr = 0; curr < hand.size(); ++curr) {
                const auto c = hand[curr];
                if (std::isdigit(c)) {
                    if (curr == hand.size() - 1) {
                        return util::err<SfenError>("piece count found at end of hand with no matching piece");
                    }

                    nextCount = *util::tryParseDigit(c);

                    const auto next = hand[curr + 1];
                    if (std::isdigit(next)) {
                        if (curr == hand.size() - 2) {
                            return util::err<SfenError>("piece count found at end of hand with no matching piece");
                        }

                        nextCount *= 10;
                        nextCount += *util::tryParseDigit(next);

                        ++curr;
                    }

                    if (nextCount == 0) {
                        return util::err<SfenError>("0 found in hand");
                    }
                } else if (const auto piece = Piece::unpromotedFromChar(hand[curr])) {
                    pos.m_hands[piece.color().idx()].set(piece.type(), nextCount);
                    nextCount = 1;
                } else {
                    return util::err<SfenError>(fmt::format("invalid piece {} found in hand", c));
                }
            }
        }

        if (sfen.size() == 4 && !util::tryParse(pos.m_moveCount, sfen[3])) {
            return util::err<SfenError>(fmt::format("invalid move count {}", sfen[3]));
        }

        pos.regenKey();
        pos.updateAttacks();

        if (pos.isInCheck()) {
            pos.m_consecutiveChecks[pos.stm().idx()] = 1;
        }

        return util::ok(pos);
    }

    util::Result<Position, SfenError> Position::fromSfen(std::string_view sfen) {
        std::vector<std::string_view> parts{};
        parts.reserve(4);

        util::split(parts, sfen);

        return fromSfenParts(parts);
    }
} // namespace stoat

fmt::format_context::iterator fmt::formatter<stoat::Position>::format(
    const stoat::Position& value,
    format_context& ctx
) const {
    using namespace stoat;

    format_to(ctx.out(), "   9   8   7   6   5   4   3   2   1\n");
    format_to(ctx.out(), " +---+---+---+---+---+---+---+---+---+\n");

    for (i32 rank = 8; rank >= 0; --rank) {
        for (i32 file = 0; file < 9; ++file) {
            const auto piece = value.pieceOn(Square::fromFileRank(file, rank));
            if (piece) {
                format_to(ctx.out(), " |{}{}", !piece.type().isPromoted() ? " " : "", piece);
            } else {
                format_to(ctx.out(), " |  ");
            }
        }

        format_to(ctx.out(), " | {}\n", static_cast<char>('a' + 8 - rank));
        format_to(ctx.out(), " +---+---+---+---+---+---+---+---+---+\n");
    }

    format_to(ctx.out(), "\n");

    format_to(ctx.out(), "Black pieces in hand: {}\n", value.hand(Colors::kBlack));
    format_to(ctx.out(), "White pieces in hand: {}\n", value.hand(Colors::kWhite));

    format_to(ctx.out(), "\n");

    format_to(ctx.out(), "{} to move", value.stm() == stoat::Colors::kBlack ? "Black" : "White");

    return ctx.out();
}

fmt::format_context::iterator fmt::formatter<stoat::Hand>::format(const stoat::Hand& value, format_context& ctx) const {
    using namespace stoat;

    bool first = true;

    const auto print = [&](PieceType pt) {
        const auto count = value.count(pt);

        if (count == 0) {
            return;
        }

        if (!first) {
            format_to(ctx.out(), " ");
        } else {
            first = false;
        }

        if (count > 1) {
            format_to(ctx.out(), "{}", count);
        }

        format_to(ctx.out(), "{}", pt);
    };

    print(PieceTypes::kRook);
    print(PieceTypes::kBishop);
    print(PieceTypes::kGold);
    print(PieceTypes::kSilver);
    print(PieceTypes::kKnight);
    print(PieceTypes::kLance);
    print(PieceTypes::kPawn);

    return ctx.out();
}
