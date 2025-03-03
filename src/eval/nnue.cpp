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

namespace {
    INCBIN(std::byte, defaultNet, ST_NETWORK_FILE);
}

namespace stoat::eval::nnue {
    namespace {
        struct Network {
            util::MultiArray<i16, kFtSize, kL1Size> ftWeights;
            std::array<i16, kL1Size> ftBiases;
            util::MultiArray<i16, 2, kL1Size> l1Weights;
            i16 l1Bias;
        };

        const Network& s_network = *reinterpret_cast<const Network*>(g_defaultNetData);

        [[nodiscard]] i32 forward(const Accumulator& acc, Color stm) {
            const auto screlu = [](i16 v) {
                const auto clipped = std::clamp(static_cast<i32>(v), 0, kFtQ);
                return clipped * clipped;
            };

            const std::span stmAccum = stm == Colors::kBlack ? acc.black : acc.white;
            const std::span nstmAccum = stm == Colors::kBlack ? acc.white : acc.black;

            i32 out = 0;

            for (u32 i = 0; i < kL1Size; ++i) {
                out += screlu(stmAccum[i]) * s_network.l1Weights[0][i];
                out += screlu(nstmAccum[i]) * s_network.l1Weights[1][i];
            }

            out /= kFtQ;
            out += s_network.l1Bias;

            return out * kScale / (kFtQ * kL1Q);
        }

        inline void addSub(std::span<const i16, kL1Size> src, std::span<i16, kL1Size> dst, u32 add, u32 sub) {
            for (u32 i = 0; i < kL1Size; ++i) {
                dst[i] = src[i] + s_network.ftWeights[add][i] - s_network.ftWeights[sub][i];
            }
        }

        inline void addAddSubSub(
            std::span<const i16, kL1Size> src,
            std::span<i16, kL1Size> dst,
            u32 add1,
            u32 add2,
            u32 sub1,
            u32 sub2
        ) {
            for (u32 i = 0; i < kL1Size; ++i) {
                dst[i] = src[i] + s_network.ftWeights[add1][i] - s_network.ftWeights[sub1][i]
                       + s_network.ftWeights[add2][i] - s_network.ftWeights[sub2][i];
            }
        }

        void applyUpdates(const NnueUpdates& updates, const Accumulator& src, Accumulator& dst) {
            const auto addCount = updates.adds.size();
            const auto subCount = updates.subs.size();

            if (addCount == 1 && subCount == 1) {
                const auto [blackAdd, whiteAdd] = updates.adds[0];
                const auto [blackSub, whiteSub] = updates.subs[0];
                addSub(src.black, dst.black, blackAdd, blackSub);
                addSub(src.white, dst.white, whiteAdd, whiteSub);
            } else if (addCount == 2 && subCount == 2) {
                const auto [blackAdd1, whiteAdd1] = updates.adds[0];
                const auto [blackAdd2, whiteAdd2] = updates.adds[1];
                const auto [blackSub1, whiteSub1] = updates.subs[0];
                const auto [blackSub2, whiteSub2] = updates.subs[1];
                addAddSubSub(src.black, dst.black, blackAdd1, blackAdd2, blackSub1, blackSub2);
                addAddSubSub(src.white, dst.white, whiteAdd1, whiteAdd2, whiteSub1, whiteSub2);
            } else {
                std::cerr << "??" << std::endl;
                assert(false);
                std::terminate();
            }
        }
    } // namespace

    void Accumulator::activate(u32 blackFeature, u32 whiteFeature) {
        for (u32 i = 0; i < kL1Size; ++i) {
            black[i] += s_network.ftWeights[blackFeature][i];
        }

        for (u32 i = 0; i < kL1Size; ++i) {
            white[i] += s_network.ftWeights[whiteFeature][i];
        }
    }

    void Accumulator::deactivate(u32 blackFeature, u32 whiteFeature) {
        for (u32 i = 0; i < kL1Size; ++i) {
            black[i] -= s_network.ftWeights[blackFeature][i];
        }

        for (u32 i = 0; i < kL1Size; ++i) {
            white[i] -= s_network.ftWeights[whiteFeature][i];
        }
    }

    void Accumulator::reset(const Position& pos) {
        std::ranges::copy(s_network.ftBiases, black.begin());
        std::ranges::copy(s_network.ftBiases, white.begin());

        auto occ = pos.occupancy();
        while (!occ.empty()) {
            const auto sq = occ.popLsb();
            const auto piece = pos.pieceOn(sq);

            const auto blackFeature = psqtFeatureIndex(Colors::kBlack, piece, sq);
            const auto whiteFeature = psqtFeatureIndex(Colors::kWhite, piece, sq);
            activate(blackFeature, whiteFeature);
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
                    for (u32 featureCount = 0; featureCount < count; ++featureCount) {
                        const auto blackFeature = handFeatureIndex(Colors::kBlack, pt, c, featureCount);
                        const auto whiteFeature = handFeatureIndex(Colors::kWhite, pt, c, featureCount);
                        activate(blackFeature, whiteFeature);
                    }
                }
            }
        };

        activateHand(Colors::kBlack);
        activateHand(Colors::kWhite);
    }

    NnueState::NnueState() {
        m_accStacc.resize(kMaxDepth + 1);
    }

    void NnueState::reset(const Position& pos) {
        m_curr = &m_accStacc[0];
        m_curr->reset(pos);
    }

    void NnueState::push(const NnueUpdates& updates) {
        assert(m_curr < &m_accStacc[kMaxDepth]);
        auto next = m_curr + 1;
        applyUpdates(updates, *m_curr, *next);
        m_curr = next;
    }

    void NnueState::pop() {
        assert(m_curr > &m_accStacc[0]);
        --m_curr;
    }

    void NnueState::applyInPlace(const NnueUpdates& updates) {
        assert(m_curr);
        applyUpdates(updates, *m_curr, *m_curr);
    }

    i32 NnueState::evaluate(Color stm) const {
        assert(m_curr);
        return forward(*m_curr, stm);
    }

    i32 evaluateOnce(const Position& pos) {
        Accumulator acc{};
        acc.reset(pos);
        return forward(acc, pos.stm());
    }
} // namespace stoat::eval::nnue
