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

        inline void pushMove(KingPair kings, Piece src, Piece dst, Square from, Square to) {
            assert(src);
            assert(dst);
            assert(from);
            assert(to);
            assert(src.color() == dst.color());

            const auto blackSrcFeature = psqtFeatureIndex(Colors::kBlack, kings, src, from);
            const auto whiteSrcFeature = psqtFeatureIndex(Colors::kWhite, kings, src, from);
            subs.push({blackSrcFeature, whiteSrcFeature});

            const auto blackDstFeature = psqtFeatureIndex(Colors::kBlack, kings, dst, to);
            const auto whiteDstFeature = psqtFeatureIndex(Colors::kWhite, kings, dst, to);
            adds.push({blackDstFeature, whiteDstFeature});
        }

        inline void pushCapture(KingPair kings, Square sq, Piece captured, u32 currHandCount) {
            assert(sq);
            assert(captured);

            const auto blackCapturedFeature = psqtFeatureIndex(Colors::kBlack, kings, captured, sq);
            const auto whiteCapturedFeature = psqtFeatureIndex(Colors::kWhite, kings, captured, sq);
            subs.push({blackCapturedFeature, whiteCapturedFeature});

            const auto blackHandFeature =
                handFeatureIndex(Colors::kBlack, captured.type().unpromoted(), captured.color().flip(), currHandCount);
            const auto whiteHandFeature =
                handFeatureIndex(Colors::kWhite, captured.type().unpromoted(), captured.color().flip(), currHandCount);
            adds.push({blackHandFeature, whiteHandFeature});
        }

        inline void pushDrop(KingPair kings, Piece piece, Square to, u32 currHandCount) {
            assert(piece);
            assert(to);
            assert(currHandCount > 0);

            const auto blackDroppedFeature = psqtFeatureIndex(Colors::kBlack, kings, piece, to);
            const auto whiteDroppedFeature = psqtFeatureIndex(Colors::kWhite, kings, piece, to);
            adds.push({blackDroppedFeature, whiteDroppedFeature});

            const auto blackHandFeature =
                handFeatureIndex(Colors::kBlack, piece.type(), piece.color(), currHandCount - 1);
            const auto whiteHandFeature =
                handFeatureIndex(Colors::kWhite, piece.type(), piece.color(), currHandCount - 1);
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

    struct SingleAccumulator {
        alignas(64) std::array<i16, kL1Size> values;
    };

    struct Accumulator {
        std::array<SingleAccumulator, 2> accs;

        [[nodiscard]] std::array<i16, kL1Size>& black() {
            return accs[Colors::kBlack.idx()].values;
        }

        [[nodiscard]] std::array<i16, kL1Size>& white() {
            return accs[Colors::kWhite.idx()].values;
        }

        [[nodiscard]] std::array<i16, kL1Size>& color(Color c) {
            assert(c);
            return accs[c.idx()].values;
        }

        [[nodiscard]] const std::array<i16, kL1Size>& black() const {
            return accs[Colors::kBlack.idx()].values;
        }

        [[nodiscard]] const std::array<i16, kL1Size>& white() const {
            return accs[Colors::kWhite.idx()].values;
        }

        [[nodiscard]] const std::array<i16, kL1Size>& color(Color c) const {
            assert(c);
            return accs[c.idx()].values;
        }

        void activate(Color c, u32 feature);
        void activate(u32 blackFeature, u32 whiteFeature);

        void reset(const Position& pos, Color c);
        void reset(const Position& pos);
    };

    class NnueState {
    public:
        NnueState();

        void reset(const Position& pos);

        void push(const Position& pos, const NnueUpdates& updates);
        void pop();

        void applyInPlace(const Position& pos, const NnueUpdates& updates);

        [[nodiscard]] i32 evaluate(Color stm) const;

    private:
        std::vector<Accumulator> m_accStacc{};
        Accumulator* m_curr{nullptr};
    };

    [[nodiscard]] i32 evaluateOnce(const Position& pos);

    [[nodiscard]] constexpr bool requiresRefresh(Color c, Square kingSq, Square prevKingSq) {
        assert(prevKingSq);
        assert(kingSq);

        const bool flip = kingSq.relative(c).file() > 4;
        const bool prevFlip = prevKingSq.relative(c).file() > 4;

        return flip != prevFlip;
    }
} // namespace stoat::eval::nnue
