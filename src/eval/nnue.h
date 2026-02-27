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

#include <array>
#include <cassert>
#include <limits>
#include <utility>
#include <vector>

#include "../core.h"
#include "../position.h"
#include "../util/static_vector.h"
#include "arch.h"

namespace stoat::eval::nnue {
    constexpr u32 kHandFeatures = 38;

    constexpr u32 kPieceStride = Squares::kCount;
    constexpr u32 kHandOffset = kPieceStride * PieceTypes::kCount;
    constexpr u32 kColorStride = kHandOffset + kHandFeatures;

    [[nodiscard]] constexpr Square transformRelativeSquare(Square kingSq, Square sq) {
        return kingSq.file() > 4 ? sq.flipFile() : sq;
    }

    [[nodiscard]] constexpr u32 psqtFeatureIndex(Color perspective, KingPair kings, Piece piece, Square sq) {
        sq = sq.relative(perspective);
        sq = transformRelativeSquare(kings.relativeKingSq(perspective), sq);
        return kColorStride * (piece.color() != perspective) + kPieceStride * piece.type().idx() + sq.idx();
    }

    [[nodiscard]] constexpr u32 handFeatureIndex(Color perspective, PieceType pt, Color handColor, u32 countMinusOne) {
        constexpr std::array kPieceOffsets = {0, 18, 22, 26, 30, 32, 34};

        return kColorStride * (handColor != perspective) + kHandOffset + kPieceOffsets[pt.idx()] + countMinusOne;
    }

    struct NnueUpdates {
        using Update = std::array<u32, 2>;

        std::array<bool, 2> refresh{};

        util::StaticVector<Update, 2> adds{};
        util::StaticVector<Update, 2> subs{};

        inline void pushPieceAdded(KingPair kings, Piece piece, Square sq) {
            const auto blackFeature = psqtFeatureIndex(Colors::kBlack, kings, piece, sq);
            const auto whiteFeature = psqtFeatureIndex(Colors::kWhite, kings, piece, sq);
            adds.push({blackFeature, whiteFeature});
        }

        inline void pushPieceRemoved(KingPair kings, Piece piece, Square sq) {
            const auto blackFeature = psqtFeatureIndex(Colors::kBlack, kings, piece, sq);
            const auto whiteFeature = psqtFeatureIndex(Colors::kWhite, kings, piece, sq);
            subs.push({blackFeature, whiteFeature});
        }

        inline void pushHandIncrement(Color c, PieceType pt, u32 countAfter) {
            assert(c);
            assert(pt);
            assert(countAfter > 0);

            const auto blackHandFeature = handFeatureIndex(Colors::kBlack, pt, c, countAfter - 1);
            const auto whiteHandFeature = handFeatureIndex(Colors::kWhite, pt, c, countAfter - 1);

            adds.push({blackHandFeature, whiteHandFeature});
        }

        inline void pushHandDecrement(Color c, PieceType pt, u32 countAfter) {
            assert(c);
            assert(pt);
            assert(countAfter < maxPiecesInHand(pt));

            const auto blackHandFeature = handFeatureIndex(Colors::kBlack, pt, c, countAfter);
            const auto whiteHandFeature = handFeatureIndex(Colors::kWhite, pt, c, countAfter);

            subs.push({blackHandFeature, whiteHandFeature});
        }

        inline void setRefresh(Color c) {
            assert(c);
            refresh[c.idx()] = true;
        }

        [[nodiscard]] inline bool requiresRefresh(Color c) const {
            assert(c);
            return refresh[c.idx()];
        }
    };

    struct UpdateContext {
        NnueUpdates updates{};
        KingPair kingSquares{};
    };

    struct BoardObserver {
        UpdateContext& ctx;

        void prepareKingMove(Color c, Square src, Square dst);

        void pieceAdded(const Position& pos, Piece piece, Square sq);
        void pieceRemoved(const Position& pos, Piece piece, Square sq);
        void pieceMutated(const Position& pos, Piece oldPiece, Piece newPiece, Square sq);
        void pieceMoved(const Position& pos, Piece piece, Square src, Square dst);
        void piecePromoted(const Position& pos, Piece oldPiece, Square src, Piece newPiece, Square dst);

        void pieceAddedToHand(Color c, PieceType pt, u32 countAfter);
        void pieceRemovedFromHand(Color c, PieceType pt, u32 countAfter);

        void finalize(const Position& pos);
    };

    struct SingleAccumulator {
        alignas(64) std::array<i16, kL1Size> values;
    };

    struct Accumulator {
        std::array<SingleAccumulator, 2> accs;

        [[nodiscard]] std::span<i16, kL1Size> black() {
            return accs[Colors::kBlack.idx()].values;
        }

        [[nodiscard]] std::span<i16, kL1Size> white() {
            return accs[Colors::kWhite.idx()].values;
        }

        [[nodiscard]] std::span<i16, kL1Size> color(Color c) {
            assert(c);
            return accs[c.idx()].values;
        }

        [[nodiscard]] std::span<const i16, kL1Size> black() const {
            return accs[Colors::kBlack.idx()].values;
        }

        [[nodiscard]] std::span<const i16, kL1Size> white() const {
            return accs[Colors::kWhite.idx()].values;
        }

        [[nodiscard]] std::span<const i16, kL1Size> color(Color c) const {
            assert(c);
            return accs[c.idx()].values;
        }

        void activate(Color c, u32 feature);
        void activate(u32 blackFeature, u32 whiteFeature);

        void reset(const Position& pos, Color c);
        void reset(const Position& pos);
    };

    struct UpdatableAccumulator {
        Accumulator acc{};
        UpdateContext ctx{};
        std::array<bool, 2> dirty{};

        inline void setDirty() {
            dirty[0] = dirty[1] = true;
        }

        inline void setUpdated(Color c) {
            assert(c != Colors::kNone);
            dirty[c.idx()] = false;
        }

        [[nodiscard]] inline bool isDirty(Color c) const {
            assert(c != Colors::kNone);
            return dirty[c.idx()];
        }
    };

    class NnueState {
    public:
        NnueState();

        void reset(const Position& pos);

        BoardObserver push();
        void pop();

        void applyImmediately(const UpdateContext& ctx, const Position& pos);

        [[nodiscard]] i32 evaluate(const Position& pos);

    private:
        std::vector<UpdatableAccumulator> m_accStacc{};
        UpdatableAccumulator* m_top{nullptr};

        void ensureUpToDate(const Position& pos);
    };

    [[nodiscard]] i32 evaluateOnce(const Position& pos);

    [[nodiscard]] constexpr bool requiresRefresh(Color c, Square kingSq, Square prevKingSq) {
        assert(prevKingSq);
        assert(kingSq);

        const bool flip = kingSq.relative(c).file() > 4;
        const bool prevFlip = prevKingSq.relative(c).file() > 4;

        return flip != prevFlip;
    }

    inline void BoardObserver::prepareKingMove(Color c, Square src, Square dst) {
        if (requiresRefresh(c, dst, src)) {
            ctx.updates.setRefresh(c);
        }
    }

    inline void BoardObserver::pieceAdded(const Position& pos, Piece piece, Square sq) {
        ctx.updates.pushPieceAdded(pos.kingSquares(), piece, sq);
    }

    inline void BoardObserver::pieceRemoved(const Position& pos, Piece piece, Square sq) {
        ctx.updates.pushPieceRemoved(pos.kingSquares(), piece, sq);
    }

    inline void BoardObserver::pieceMutated(const Position& pos, Piece oldPiece, Piece newPiece, Square sq) {
        ctx.updates.pushPieceRemoved(pos.kingSquares(), oldPiece, sq);
        ctx.updates.pushPieceAdded(pos.kingSquares(), newPiece, sq);
    }

    inline void BoardObserver::pieceMoved(const Position& pos, Piece piece, Square src, Square dst) {
        ctx.updates.pushPieceRemoved(pos.kingSquares(), piece, src);
        ctx.updates.pushPieceAdded(pos.kingSquares(), piece, dst);
    }

    inline void BoardObserver::piecePromoted(
        const Position& pos,
        Piece oldPiece,
        Square src,
        Piece newPiece,
        Square dst
    ) {
        ctx.updates.pushPieceRemoved(pos.kingSquares(), oldPiece, src);
        ctx.updates.pushPieceAdded(pos.kingSquares(), newPiece, dst);
    }

    inline void BoardObserver::pieceAddedToHand(Color c, PieceType pt, u32 countAfter) {
        ctx.updates.pushHandIncrement(c, pt, countAfter);
    }

    inline void BoardObserver::pieceRemovedFromHand(Color c, PieceType pt, u32 countAfter) {
        ctx.updates.pushHandDecrement(c, pt, countAfter);
    }

    inline void BoardObserver::finalize(const Position& pos) {
        ctx.kingSquares = pos.kingSquares();
    }
} // namespace stoat::eval::nnue
