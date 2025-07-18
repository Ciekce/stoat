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

#include "../util/multi_array.h"

namespace {
    INCBIN(std::byte, defaultNet, ST_NETWORK_FILE);
}

namespace stoat::eval::nnue {
    namespace {
        // Stoat lacks output buckets, so imagine that there is a
        // `kOutputBuckets` dimension on each of these sets of params
        //   - including l3Bias, it's the output bias, same as a singlelayer net
        // Ideas welcome regarding what to output bucket a shogi net on
        struct Network {
            alignas(64) util::MultiArray<i16, kFtSize, kL1Size> ftWeights;
            alignas(64) util::MultiArray<i16, kL1Size> ftBiases;
            alignas(64) util::MultiArray<i8, kL1Size * kL2Size> l1Weights;
            alignas(64) util::MultiArray<i32, kL2Size> l1Biases;
            alignas(64) util::MultiArray<i32, kL2Size * 2, kL3Size> l2Weights;
            alignas(64) util::MultiArray<i32, kL3Size> l2Biases;
            alignas(64) util::MultiArray<i32, kL3Size> l3Weights;
            alignas(64) i32 l3Bias;
        };

        const Network& s_network = *reinterpret_cast<const Network*>(g_defaultNetData);

        [[nodiscard]] i32 forward(const Accumulator& acc, Color stm) {
            static constexpr auto kChunkSize8 = sizeof(__m256i) / sizeof(i8);
            static constexpr auto kChunkSize16 = sizeof(__m256i) / sizeof(i16);
            static constexpr auto kChunkSize32 = sizeof(__m256i) / sizeof(i32);

            static constexpr auto k32ChunkSize8 = sizeof(i32) / sizeof(u8);

            static constexpr auto kPairCount = kL1Size / 2;

            // Overall shift to simultaneously undo FT shift, dequantise
            // from FT and L1 Qs, and requantise with later layer Q
            static constexpr auto kL1Shift = 16 + kQBits - kFtScaleBits - kFtQBits - kFtQBits - kL1QBits;

            static constexpr i32 kQ = 1 << kQBits;

            // Activated FT outputs (concated accumulators)
            alignas(64) std::array<u8, kL1Size> ftOut;
            // Activated L1 outputs (dual activation, hence doubled size)
            alignas(64) std::array<i32, kL2Size * 2> l1Out;
            // *UN*activated L2 outputs
            alignas(64) std::array<i32, kL3Size> l2Out;

            // activate FT - pairwise crelu
            // See cj+shawnpasta in SF for explanation, this is a simple scalar translation of the same logic
            const auto activatePerspective = [&](std::span<const i16, kL1Size> inputs, usize outputOffset) {
                for (usize inputIdx = 0; inputIdx < kPairCount; ++inputIdx) {
                    auto i1 = inputs[inputIdx];
                    auto i2 = inputs[inputIdx + kPairCount];

                    // crelu both sides
                    // skip the max for the second element of the pair
                    i1 = std::clamp<i16>(i1, 0, (1 << kFtQBits) - 1);
                    i2 = std::min<i16>(i2, (1 << kFtQBits) - 1);

                    const i16 s = i1 << kFtScaleBits;

                    // mulhi at home (exactly the same trick as in TT indexing - widen, mul, shift top bits down)
                    // the casts are redundant because of C++ integer promotion, but included for clarity
                    const auto p = (static_cast<i32>(s) * static_cast<i32>(i2)) >> 16;

                    // saturate to u8 range before downcasting (packus_epi16 does this itself)
                    const auto packed = static_cast<u8>(std::clamp(p, 0, 255));

                    ftOut[outputOffset + inputIdx] = packed;
                }
            };

            // activate STM accumulator into ftOut[0..L1/2]
            activatePerspective(acc.color(stm), 0);
            // activate NSTM accumulator into ftOut[L1/2..L2]
            activatePerspective(acc.color(stm.flip()), kPairCount);

            // Per the above comment regarding output buckets, imagine that every
            // single index into a weight or bias array is first indexed by output bucket

            // Unactivated L1 outputs in FtQ*L1Q space
            std::array<i32, kL2Size> intermediate{};

            // perform L1 matmul
            for (usize inputIdx = 0; inputIdx < kL1Size; ++inputIdx) {
                const auto i = ftOut[inputIdx];

                for (usize outputIdx = 0; outputIdx < kL2Size; ++outputIdx) {
                    // pretend this is just `inputIdx * kL2Size + outputIdx` or even simply [inputIdx][outputIdx]
                    // the way dpbusd works requires this weight ordering and I'm not repermuting the net for ts
                    const auto weightIdx = (inputIdx - (inputIdx % 4)) * kL2Size + outputIdx * 4 + (inputIdx % 4);
                    const auto w = s_network.l1Weights[weightIdx];
                    intermediate[outputIdx] += i * w;
                }
            }

            // requantise, add biases and activate L1 outputs
            for (usize i = 0; i < kL2Size; ++i) {
                const auto bias = s_network.l1Biases[i];

                auto out = intermediate[i];

                // requantise to later layer Q + undo ft shift in one go
                // (this is ultimately a shift down, expressed as a
                // negative shift up, so negate the actual shift amount)
                out >>= -kL1Shift;

                out += bias;

                auto crelu = out;
                auto screlu = out;

                // relu + clip
                crelu = std::clamp(crelu, 0, kQ);
                // shift into Q*Q space (currently Q) to match squared side
                crelu <<= kQBits;

                screlu *= screlu;
                // clip in Q*Q space (we just squared this value, so we squared Q too)
                screlu = std::min(screlu, kQ * kQ);

                l1Out[i] = crelu;
                l1Out[i + kL2Size] = screlu;
            }

            // values are now in Q*Q space (see above)

            std::ranges::copy(s_network.l2Biases, l2Out.begin());

            // perform L2 matmul
            for (usize inputIdx = 0; inputIdx < kL2Size * 2; ++inputIdx) {
                const auto input = l1Out[inputIdx];

                for (usize outputIdx = 0; outputIdx < kL3Size; ++outputIdx) {
                    const auto w = s_network.l2Weights[inputIdx][outputIdx];
                    l2Out[outputIdx] += input * w;
                }
            }

            // values are now in Q*Q*Q space, we just multiplied Q*Q values by Q weights

            i32 out = s_network.l3Bias;

            // activate L2 outputs and perform L3 matmul
            for (usize inputIdx = 0; inputIdx < kL3Size; ++inputIdx) {
                auto i = l2Out[inputIdx];
                const auto w = s_network.l3Weights[inputIdx];

                // crelu
                i = std::clamp(i, 0, kQ * kQ * kQ);

                out += i * w;
            }

            // values are now in Q*Q*Q*Q space

            // dequantise by one step before scaling to avoid overflow
            out /= kQ;
            out *= kScale;
            // dequantise the rest of the way
            out /= kQ * kQ * kQ;

            return out;
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

        void applyUpdates(const Position& pos, const NnueUpdates& updates, const Accumulator& src, Accumulator& dst) {
            const auto addCount = updates.adds.size();
            const auto subCount = updates.subs.size();

            for (const auto c : {Colors::kBlack, Colors::kWhite}) {
                if (updates.requiresRefresh(c)) {
                    dst.reset(pos, c);
                    continue;
                }

                if (addCount == 1 && subCount == 1) {
                    const auto add = updates.adds[0][c.idx()];
                    const auto sub = updates.subs[0][c.idx()];
                    addSub(src.color(c), dst.color(c), add, sub);
                } else if (addCount == 2 && subCount == 2) {
                    const auto add1 = updates.adds[0][c.idx()];
                    const auto add2 = updates.adds[1][c.idx()];
                    const auto sub1 = updates.subs[0][c.idx()];
                    const auto sub2 = updates.subs[1][c.idx()];
                    addAddSubSub(src.color(c), dst.color(c), add1, add2, sub1, sub2);
                } else {
                    fmt::println(stderr, "??");
                    assert(false);
                    std::terminate();
                }
            }
        }
    } // namespace

    void Accumulator::activate(Color c, u32 feature) {
        auto& acc = color(c);
        for (u32 i = 0; i < kL1Size; ++i) {
            acc[i] += s_network.ftWeights[feature][i];
        }
    }

    void Accumulator::activate(u32 blackFeature, u32 whiteFeature) {
        auto& black = this->black();
        auto& white = this->white();

        for (u32 i = 0; i < kL1Size; ++i) {
            black[i] += s_network.ftWeights[blackFeature][i];
        }

        for (u32 i = 0; i < kL1Size; ++i) {
            white[i] += s_network.ftWeights[whiteFeature][i];
        }
    }

    void Accumulator::reset(const Position& pos, Color c) {
        std::ranges::copy(s_network.ftBiases, color(c).begin());

        const auto kings = pos.kingSquares();

        auto occ = pos.occupancy();
        while (!occ.empty()) {
            const auto sq = occ.popLsb();
            const auto piece = pos.pieceOn(sq);

            const auto feature = psqtFeatureIndex(c, kings, piece, sq);
            activate(c, feature);
        }

        const auto activateHand = [&](Color handColor) {
            const auto& hand = pos.hand(handColor);

            if (hand.empty()) {
                return;
            }

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
                for (u32 featureCount = 0; featureCount < count; ++featureCount) {
                    const auto feature = handFeatureIndex(c, pt, handColor, featureCount);
                    activate(c, feature);
                }
            }
        };

        activateHand(Colors::kBlack);
        activateHand(Colors::kWhite);
    }

    void Accumulator::reset(const Position& pos) {
        std::ranges::copy(s_network.ftBiases, black().begin());
        std::ranges::copy(s_network.ftBiases, white().begin());

        const auto kings = pos.kingSquares();

        auto occ = pos.occupancy();
        while (!occ.empty()) {
            const auto sq = occ.popLsb();
            const auto piece = pos.pieceOn(sq);

            const auto blackFeature = psqtFeatureIndex(Colors::kBlack, kings, piece, sq);
            const auto whiteFeature = psqtFeatureIndex(Colors::kWhite, kings, piece, sq);
            activate(blackFeature, whiteFeature);
        }

        const auto activateHand = [&](Color c) {
            const auto& hand = pos.hand(c);

            if (hand.empty()) {
                return;
            }

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
                for (u32 featureCount = 0; featureCount < count; ++featureCount) {
                    const auto blackFeature = handFeatureIndex(Colors::kBlack, pt, c, featureCount);
                    const auto whiteFeature = handFeatureIndex(Colors::kWhite, pt, c, featureCount);
                    activate(blackFeature, whiteFeature);
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

    void NnueState::push(const Position& pos, const NnueUpdates& updates) {
        assert(m_curr < &m_accStacc[kMaxDepth]);
        auto next = m_curr + 1;
        applyUpdates(pos, updates, *m_curr, *next);
        m_curr = next;
    }

    void NnueState::pop() {
        assert(m_curr > &m_accStacc[0]);
        --m_curr;
    }

    void NnueState::applyInPlace(const Position& pos, const NnueUpdates& updates) {
        assert(m_curr);
        applyUpdates(pos, updates, *m_curr, *m_curr);
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
