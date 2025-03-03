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

    [[nodiscard]] constexpr u32 psqtFeatureIndex(Color perspective, Piece piece, Square sq) {
        if (perspective == Colors::kWhite) {
            sq = sq.rotate();
        }
        return kColorStride * (piece.color() != perspective) + kPieceStride * piece.type().idx() + sq.idx();
    }

    [[nodiscard]] constexpr u32 handFeatureIndex(Color perspective, PieceType pt, Color handColor, u32 countMinusOne) {
        constexpr auto kPieceOffsets = [] {
            std::array<u32, PieceTypes::kCount> offsets{};
            offsets.fill(std::numeric_limits<u32>::max());

            offsets[PieceTypes::kPawn.idx()] = 0;
            offsets[PieceTypes::kLance.idx()] = 18;
            offsets[PieceTypes::kKnight.idx()] = 22;
            offsets[PieceTypes::kSilver.idx()] = 26;
            offsets[PieceTypes::kGold.idx()] = 30;
            offsets[PieceTypes::kBishop.idx()] = 34;
            offsets[PieceTypes::kRook.idx()] = 36;

            return offsets;
        }();

        return kColorStride * (handColor != perspective) + kHandOffset + kPieceOffsets[pt.idx()] + countMinusOne;
    }

    struct NnueUpdates {
        using Update = std::pair<u32, u32>;

        util::StaticVector<Update, 2> adds{};
        util::StaticVector<Update, 2> subs{};

        inline void pushMove(Piece src, Piece dst, Square from, Square to) {
            assert(src);
            assert(dst);
            assert(from);
            assert(to);
            assert(src.color() == dst.color());

            const auto blackSrcFeature = psqtFeatureIndex(Colors::kBlack, src, from);
            const auto whiteSrcFeature = psqtFeatureIndex(Colors::kWhite, src, from);
            subs.push({blackSrcFeature, whiteSrcFeature});

            const auto blackDstFeature = psqtFeatureIndex(Colors::kBlack, dst, to);
            const auto whiteDstFeature = psqtFeatureIndex(Colors::kWhite, dst, to);
            adds.push({blackDstFeature, whiteDstFeature});
        }

        inline void pushCapture(Square sq, Piece captured, u32 currHandCount) {
            assert(sq);
            assert(captured);

            const auto blackCapturedFeature = psqtFeatureIndex(Colors::kBlack, captured, sq);
            const auto whiteCapturedFeature = psqtFeatureIndex(Colors::kWhite, captured, sq);
            subs.push({blackCapturedFeature, whiteCapturedFeature});

            const auto blackHandFeature =
                handFeatureIndex(Colors::kBlack, captured.type().unpromoted(), captured.color().flip(), currHandCount);
            const auto whiteHandFeature =
                handFeatureIndex(Colors::kWhite, captured.type().unpromoted(), captured.color().flip(), currHandCount);
            adds.push({blackHandFeature, whiteHandFeature});
        }

        inline void pushDrop(Piece piece, Square to, u32 currHandCount) {
            assert(piece);
            assert(to);
            assert(currHandCount > 0);

            const auto blackDroppedFeature = psqtFeatureIndex(Colors::kBlack, piece, to);
            const auto whiteDroppedFeature = psqtFeatureIndex(Colors::kWhite, piece, to);
            adds.push({blackDroppedFeature, whiteDroppedFeature});

            const auto blackHandFeature =
                handFeatureIndex(Colors::kBlack, piece.type(), piece.color(), currHandCount - 1);
            const auto whiteHandFeature =
                handFeatureIndex(Colors::kWhite, piece.type(), piece.color(), currHandCount - 1);
            subs.push({blackHandFeature, whiteHandFeature});
        }
    };

    struct Accumulator {
        alignas(64) std::array<i16, kL1Size> black;
        alignas(64) std::array<i16, kL1Size> white;

        void activate(u32 blackFeature, u32 whiteFeature);
        void deactivate(u32 blackFeature, u32 whiteFeature);

        void reset(const Position& pos);
    };

    class NnueState {
    public:
        NnueState();

        void reset(const Position& pos);

        void push(const NnueUpdates& updates);
        void pop();

        void applyInPlace(const NnueUpdates& updates);

        [[nodiscard]] i32 evaluate(Color stm) const;

    private:
        std::vector<Accumulator> m_accStacc{};
        Accumulator* m_curr{nullptr};
    };

    [[nodiscard]] i32 evaluateOnce(const Position& pos);
} // namespace stoat::eval::nnue
