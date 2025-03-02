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

#include "nnue.h"

#include <algorithm>
#include <limits>

#ifdef _MSC_VER
    #define ST_MSVC
    #pragma push_macro("_MSC_VER")
    #undef _MSC_VER
#endif

#define INCBIN_PREFIX g_
#include "../3rdparty/incbin.h"

#ifdef ST_MSVC
    #pragma pop_macro("_MSC_VER")
    #undef ST_MSVC
#endif

#include "../core.h"
#include "../util/multi_array.h"
#include "arch.h"

namespace {
    INCBIN(std::byte, defaultNet, ST_NETWORK_FILE);
}

namespace stoat::eval::nnue {
    namespace {
        constexpr u32 kHandFeatures = 38;

        constexpr u32 kPieceStride = Squares::kCount;
        constexpr u32 kHandOffset = kPieceStride * PieceTypes::kCount;
        constexpr u32 kColorStride = kHandOffset + kHandFeatures;

        [[nodiscard]] u32 psqtFeatureIndex(Color perspective, Piece piece, Square sq) {
            if (perspective == Colors::kWhite) {
                sq = sq.rotate();
            }
            return kColorStride * (piece.color() != perspective) + kPieceStride * piece.type().idx() + sq.idx();
        }

        [[nodiscard]] u32 handFeatureIndex(Color perspective, Piece piece, u32 countMinusOne) {
            static constexpr auto kPieceOffsets = [] {
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

            return kColorStride * (piece.color() != perspective) + kHandOffset + kPieceOffsets[piece.type().idx()]
                 + countMinusOne;
        }

        struct Network {
            util::MultiArray<i16, kFtSize, kL1Size> ftWeights;
            std::array<i16, kL1Size> ftBiases;
            util::MultiArray<i16, 2, kL1Size> l1Weights;
            i16 l1Bias;
        };

        const Network& s_network = *reinterpret_cast<const Network*>(g_defaultNetData);
    } // namespace

    i32 evaluate(const Position& pos) {
        const auto activate = [](std::span<i16, kL1Size> accum, u32 feature) {
            for (u32 i = 0; i < kL1Size; ++i) {
                accum[i] += s_network.ftWeights[feature][i];
            }
        };

        std::array<i16, kL1Size> blackAccum{};
        std::array<i16, kL1Size> whiteAccum{};

        std::ranges::copy(s_network.ftBiases, blackAccum.begin());
        std::ranges::copy(s_network.ftBiases, whiteAccum.begin());

        auto occ = pos.occupancy();
        while (!occ.empty()) {
            const auto sq = occ.popLsb();
            const auto piece = pos.pieceOn(sq);

            const auto blackFeature = psqtFeatureIndex(Colors::kBlack, piece, sq);
            const auto whiteFeature = psqtFeatureIndex(Colors::kWhite, piece, sq);

            activate(blackAccum, blackFeature);
            activate(whiteAccum, whiteFeature);
        }

        const auto activateHand = [&](Color c) {
            const auto& hand = pos.hand(c);
            for (const auto pt :
                 {PieceTypes::kPawn,
                  PieceTypes::kLance,
                  PieceTypes::kKnight,
                  PieceTypes::kSilver,
                  PieceTypes::kGold,
                  PieceTypes::kBishop,
                  PieceTypes::kRook})
            {
                const auto count = hand.count(pt);
                if (count > 0) {
                    const auto piece = pt.withColor(c);

                    for (u32 featureCount = 0; featureCount < count; ++featureCount) {
                        const auto blackFeature = handFeatureIndex(Colors::kBlack, piece, featureCount);
                        const auto whiteFeature = handFeatureIndex(Colors::kWhite, piece, featureCount);

                        activate(blackAccum, blackFeature);
                        activate(whiteAccum, whiteFeature);
                    }
                }
            }
        };

        activateHand(Colors::kBlack);
        activateHand(Colors::kWhite);

        const auto screlu = [](i16 v) {
            const auto clipped = std::clamp(static_cast<i32>(v), 0, kFtQ);
            return clipped * clipped;
        };

        const std::span stmAccum = pos.stm() == Colors::kBlack ? blackAccum : whiteAccum;
        const std::span nstmAccum = pos.stm() == Colors::kBlack ? whiteAccum : blackAccum;

        i32 out = 0;

        for (u32 i = 0; i < kL1Size; ++i) {
            out += screlu(stmAccum[i]) * s_network.l1Weights[0][i];
            out += screlu(nstmAccum[i]) * s_network.l1Weights[1][i];
        }

        out /= kFtQ;
        out += s_network.l1Bias;

        return out * kScale / (kFtQ * kL1Q);
    }
} // namespace stoat::eval::nnue
