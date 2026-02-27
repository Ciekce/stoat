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

#include <immintrin.h>

#ifdef _MSC_VER
    #define ST_MSVC
    #pragma push_macro("_MSC_VER")
    #undef _MSC_VER
#endif

#define INCBIN_PREFIX g_
#include "../../3rdparty/incbin.h"

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

        [[nodiscard]] inline __m256i load(const void* ptr) {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
        }

        inline void store(void* ptr, __m256i vec) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), vec);
        }

        [[nodiscard]] __m256i dpbusd(__m256i acc, __m256i u, __m256i i) {
            const auto p = _mm256_maddubs_epi16(u, i);
            const auto w = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
            return _mm256_add_epi32(acc, w);
        }

        [[nodiscard]] i32 hsum32(__m256i v) {
            const auto high128 = _mm256_extracti128_si256(v, 1);
            const auto low128 = _mm256_castsi256_si128(v);

            const auto sum128 = _mm_add_epi32(high128, low128);

            const auto high64 = _mm_unpackhi_epi64(sum128, sum128);
            const auto sum64 = _mm_add_epi32(sum128, high64);

            const auto high32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
            const auto sum32 = _mm_add_epi32(sum64, high32);

            return _mm_cvtsi128_si32(sum32);
        }

        [[nodiscard]] i32 forward(const Accumulator& acc, Color stm) {
            static constexpr auto kChunkSize8 = sizeof(__m256i) / sizeof(i8);
            static constexpr auto kChunkSize16 = sizeof(__m256i) / sizeof(i16);
            static constexpr auto kChunkSize32 = sizeof(__m256i) / sizeof(i32);

            static constexpr auto k32ChunkSize8 = sizeof(i32) / sizeof(u8);

            static constexpr auto kPairCount = kL1Size / 2;

            static constexpr auto kL1Shift = 16 + kQBits - kFtScaleBits - kFtQBits - kFtQBits - kL1QBits;

            static constexpr i32 kQ = 1 << kQBits;

            alignas(64) std::array<u8, kL1Size> ftOut;
            alignas(64) std::array<i32, kL2Size * 2> l1Out;
            alignas(64) std::array<i32, kL3Size> l2Out;

            const auto zero = _mm256_setzero_si256();

            const auto ftOne = _mm256_set1_epi16((1 << kFtQBits) - 1);
            const auto l1CreluOne = _mm256_set1_epi32(kQ);
            const auto l1ScreluOne = _mm256_set1_epi32(kQ * kQ);
            const auto l2One = _mm256_set1_epi32(kQ * kQ * kQ);

            const auto activatePerspective = [&](std::span<const i16, kL1Size> inputs, usize outputOffset) {
                for (usize inputIdx = 0; inputIdx < kPairCount; inputIdx += kChunkSize16 * 4) {
                    auto i1_0 = load(&inputs[inputIdx + kChunkSize16 * 0]);
                    auto i1_1 = load(&inputs[inputIdx + kChunkSize16 * 1]);
                    auto i1_2 = load(&inputs[inputIdx + kChunkSize16 * 2]);
                    auto i1_3 = load(&inputs[inputIdx + kChunkSize16 * 3]);

                    auto i2_0 = load(&inputs[inputIdx + kPairCount + kChunkSize16 * 0]);
                    auto i2_1 = load(&inputs[inputIdx + kPairCount + kChunkSize16 * 1]);
                    auto i2_2 = load(&inputs[inputIdx + kPairCount + kChunkSize16 * 2]);
                    auto i2_3 = load(&inputs[inputIdx + kPairCount + kChunkSize16 * 3]);

                    i1_0 = _mm256_min_epi16(i1_0, ftOne);
                    i1_1 = _mm256_min_epi16(i1_1, ftOne);
                    i1_2 = _mm256_min_epi16(i1_2, ftOne);
                    i1_3 = _mm256_min_epi16(i1_3, ftOne);

                    i2_0 = _mm256_min_epi16(i2_0, ftOne);
                    i2_1 = _mm256_min_epi16(i2_1, ftOne);
                    i2_2 = _mm256_min_epi16(i2_2, ftOne);
                    i2_3 = _mm256_min_epi16(i2_3, ftOne);

                    i1_0 = _mm256_max_epi16(i1_0, zero);
                    i1_1 = _mm256_max_epi16(i1_1, zero);
                    i1_2 = _mm256_max_epi16(i1_2, zero);
                    i1_3 = _mm256_max_epi16(i1_3, zero);

                    const auto s_0 = _mm256_slli_epi16(i1_0, kFtScaleBits);
                    const auto s_1 = _mm256_slli_epi16(i1_1, kFtScaleBits);
                    const auto s_2 = _mm256_slli_epi16(i1_2, kFtScaleBits);
                    const auto s_3 = _mm256_slli_epi16(i1_3, kFtScaleBits);

                    const auto p_0 = _mm256_mulhi_epi16(s_0, i2_0);
                    const auto p_1 = _mm256_mulhi_epi16(s_1, i2_1);
                    const auto p_2 = _mm256_mulhi_epi16(s_2, i2_2);
                    const auto p_3 = _mm256_mulhi_epi16(s_3, i2_3);

                    auto packed_0 = _mm256_packus_epi16(p_0, p_1);
                    auto packed_1 = _mm256_packus_epi16(p_2, p_3);

                    packed_0 = _mm256_permute4x64_epi64(packed_0, _MM_SHUFFLE(3, 1, 2, 0));
                    packed_1 = _mm256_permute4x64_epi64(packed_1, _MM_SHUFFLE(3, 1, 2, 0));

                    store(&ftOut[outputOffset + inputIdx + kChunkSize8 * 0], packed_0);
                    store(&ftOut[outputOffset + inputIdx + kChunkSize8 * 1], packed_1);
                }
            };

            activatePerspective(acc.color(stm), 0);
            activatePerspective(acc.color(stm.flip()), kPairCount);

            const auto* ftOutI32s = reinterpret_cast<const i32*>(ftOut.data());

            alignas(64) util::MultiArray<__m256i, kL2Size / kChunkSize32, 4> intermediate{};

            for (usize inputIdx = 0; inputIdx < kL1Size; inputIdx += k32ChunkSize8 * 4) {
                const auto weightsStart = inputIdx * kL2Size;

                const auto i_0 = _mm256_set1_epi32(ftOutI32s[inputIdx / k32ChunkSize8 + 0]);
                const auto i_1 = _mm256_set1_epi32(ftOutI32s[inputIdx / k32ChunkSize8 + 1]);
                const auto i_2 = _mm256_set1_epi32(ftOutI32s[inputIdx / k32ChunkSize8 + 2]);
                const auto i_3 = _mm256_set1_epi32(ftOutI32s[inputIdx / k32ChunkSize8 + 3]);

                for (usize outputIdx = 0; outputIdx < kL2Size; outputIdx += kChunkSize32) {
                    auto& v = intermediate[outputIdx / kChunkSize32];

                    const auto w_0 =
                        load(&s_network.l1Weights[weightsStart + k32ChunkSize8 * (outputIdx + kL2Size * 0)]);
                    const auto w_1 =
                        load(&s_network.l1Weights[weightsStart + k32ChunkSize8 * (outputIdx + kL2Size * 1)]);
                    const auto w_2 =
                        load(&s_network.l1Weights[weightsStart + k32ChunkSize8 * (outputIdx + kL2Size * 2)]);
                    const auto w_3 =
                        load(&s_network.l1Weights[weightsStart + k32ChunkSize8 * (outputIdx + kL2Size * 3)]);

                    v[0] = dpbusd(v[0], i_0, w_0);
                    v[1] = dpbusd(v[1], i_1, w_1);
                    v[2] = dpbusd(v[2], i_2, w_2);
                    v[3] = dpbusd(v[3], i_3, w_3);
                }
            }

            for (usize i = 0; i < kL2Size; i += kChunkSize32) {
                const auto biases = load(&s_network.l1Biases[i]);

                const auto& v = intermediate[i / kChunkSize32];

                const auto sums_0 = _mm256_add_epi32(v[0], v[1]);
                const auto sums_1 = _mm256_add_epi32(v[2], v[3]);

                auto out = _mm256_add_epi32(sums_0, sums_1);

                out = _mm256_srai_epi32(out, -kL1Shift);
                out = _mm256_add_epi32(out, biases);

                auto crelu = out;
                auto screlu = out;

                crelu = _mm256_max_epi32(crelu, zero);
                crelu = _mm256_min_epi32(crelu, l1CreluOne);
                crelu = _mm256_slli_epi32(crelu, kQBits);

                screlu = _mm256_mullo_epi32(screlu, screlu);
                screlu = _mm256_min_epi32(screlu, l1ScreluOne);

                store(&l1Out[i], crelu);
                store(&l1Out[i + kL2Size], screlu);
            }

            std::ranges::copy(s_network.l2Biases, l2Out.begin());

            for (usize inputIdx = 0; inputIdx < kL2Size * 2; ++inputIdx) {
                const auto input = _mm256_set1_epi32(l1Out[inputIdx]);

                for (usize outputIdx = 0; outputIdx < kL3Size; outputIdx += kChunkSize32 * 4) {
                    const auto w_0 = load(&s_network.l2Weights[inputIdx][outputIdx + kChunkSize32 * 0]);
                    const auto w_1 = load(&s_network.l2Weights[inputIdx][outputIdx + kChunkSize32 * 1]);
                    const auto w_2 = load(&s_network.l2Weights[inputIdx][outputIdx + kChunkSize32 * 2]);
                    const auto w_3 = load(&s_network.l2Weights[inputIdx][outputIdx + kChunkSize32 * 3]);

                    auto out_0 = load(&l2Out[outputIdx + kChunkSize32 * 0]);
                    auto out_1 = load(&l2Out[outputIdx + kChunkSize32 * 1]);
                    auto out_2 = load(&l2Out[outputIdx + kChunkSize32 * 2]);
                    auto out_3 = load(&l2Out[outputIdx + kChunkSize32 * 3]);

                    const auto p_0 = _mm256_mullo_epi32(input, w_0);
                    const auto p_1 = _mm256_mullo_epi32(input, w_1);
                    const auto p_2 = _mm256_mullo_epi32(input, w_2);
                    const auto p_3 = _mm256_mullo_epi32(input, w_3);

                    out_0 = _mm256_add_epi32(out_0, p_0);
                    out_1 = _mm256_add_epi32(out_1, p_1);
                    out_2 = _mm256_add_epi32(out_2, p_2);
                    out_3 = _mm256_add_epi32(out_3, p_3);

                    store(&l2Out[outputIdx + kChunkSize32 * 0], out_0);
                    store(&l2Out[outputIdx + kChunkSize32 * 1], out_1);
                    store(&l2Out[outputIdx + kChunkSize32 * 2], out_2);
                    store(&l2Out[outputIdx + kChunkSize32 * 3], out_3);
                }
            }

            auto out_0 = zero;
            auto out_1 = zero;
            auto out_2 = zero;
            auto out_3 = zero;

            for (usize inputIdx = 0; inputIdx < kL3Size; inputIdx += kChunkSize32 * 4) {
                auto i_0 = load(&l2Out[inputIdx + kChunkSize32 * 0]);
                auto i_1 = load(&l2Out[inputIdx + kChunkSize32 * 1]);
                auto i_2 = load(&l2Out[inputIdx + kChunkSize32 * 2]);
                auto i_3 = load(&l2Out[inputIdx + kChunkSize32 * 3]);

                const auto w_0 = load(&s_network.l3Weights[inputIdx + kChunkSize32 * 0]);
                const auto w_1 = load(&s_network.l3Weights[inputIdx + kChunkSize32 * 1]);
                const auto w_2 = load(&s_network.l3Weights[inputIdx + kChunkSize32 * 2]);
                const auto w_3 = load(&s_network.l3Weights[inputIdx + kChunkSize32 * 3]);

                i_0 = _mm256_max_epi32(i_0, zero);
                i_1 = _mm256_max_epi32(i_1, zero);
                i_2 = _mm256_max_epi32(i_2, zero);
                i_3 = _mm256_max_epi32(i_3, zero);

                i_0 = _mm256_min_epi32(i_0, l2One);
                i_1 = _mm256_min_epi32(i_1, l2One);
                i_2 = _mm256_min_epi32(i_2, l2One);
                i_3 = _mm256_min_epi32(i_3, l2One);

                i_0 = _mm256_mullo_epi32(i_0, w_0);
                i_1 = _mm256_mullo_epi32(i_1, w_1);
                i_2 = _mm256_mullo_epi32(i_2, w_2);
                i_3 = _mm256_mullo_epi32(i_3, w_3);

                out_0 = _mm256_add_epi32(out_0, i_0);
                out_1 = _mm256_add_epi32(out_1, i_1);
                out_2 = _mm256_add_epi32(out_2, i_2);
                out_3 = _mm256_add_epi32(out_3, i_3);
            }

            const auto s0 = _mm256_add_epi32(out_0, out_1);
            const auto s1 = _mm256_add_epi32(out_2, out_3);

            const auto s = _mm256_add_epi32(s0, s1);

            auto out = s_network.l3Bias + hsum32(s);

            out /= kQ;
            out *= kScale;
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

        void refresh(Color c, UpdatableAccumulator& acc, const Position& pos) {
            acc.acc.reset(pos, c);
            acc.setUpdated(c);
        }

        void applyUpdates(Color c, const NnueUpdates& updates, const Accumulator& src, UpdatableAccumulator& dst) {
            const auto addCount = updates.adds.size();
            const auto subCount = updates.subs.size();

            if (addCount == 1 && subCount == 1) {
                const auto add = updates.adds[0][c.idx()];
                const auto sub = updates.subs[0][c.idx()];
                addSub(src.color(c), dst.acc.color(c), add, sub);
            } else if (addCount == 2 && subCount == 2) {
                const auto add1 = updates.adds[0][c.idx()];
                const auto add2 = updates.adds[1][c.idx()];
                const auto sub1 = updates.subs[0][c.idx()];
                const auto sub2 = updates.subs[1][c.idx()];
                addAddSubSub(src.color(c), dst.acc.color(c), add1, add2, sub1, sub2);
            } else {
                fmt::println(stderr, "??");
                assert(false);
                std::terminate();
            }

            dst.setUpdated(c);
        }
    } // namespace

    void Accumulator::activate(Color c, u32 feature) {
        auto acc = color(c);
        for (u32 i = 0; i < kL1Size; ++i) {
            acc[i] += s_network.ftWeights[feature][i];
        }
    }

    void Accumulator::activate(u32 blackFeature, u32 whiteFeature) {
        auto black = this->black();
        auto white = this->white();

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

            for (const auto pt : kHandPieces) {
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

            for (const auto pt : kHandPieces) {
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
        m_top = &m_accStacc[0];
        m_top->acc.reset(pos);
    }

    BoardObserver NnueState::push() {
        assert(m_top < &m_accStacc[kMaxDepth]);

        ++m_top;

        m_top->ctx = {};
        m_top->setDirty();

        return BoardObserver{m_top->ctx};
    }

    void NnueState::pop() {
        assert(m_top > &m_accStacc[0]);
        --m_top;
    }

    void NnueState::applyImmediately(const UpdateContext& ctx, const Position& pos) {
        assert(m_top);
        assert(m_top != &m_accStacc[0]);
        for (const auto c : {Colors::kBlack, Colors::kWhite}) {
            if (m_top->ctx.updates.requiresRefresh(c)) {
                m_top->acc.reset(pos, c);
            } else {
                applyUpdates(c, ctx.updates, m_top->acc, *m_top);
            }
            m_top->setUpdated(c);
        }
    }

    i32 NnueState::evaluate(const Position& pos) {
        assert(m_top);
        ensureUpToDate(pos);
        return forward(m_top->acc, pos.stm());
    }

    void NnueState::ensureUpToDate(const Position& pos) {
        for (const auto c : {Colors::kBlack, Colors::kWhite}) {
            if (!m_top->isDirty(c)) {
                continue;
            }

            if (m_top->ctx.updates.requiresRefresh(c)) {
                refresh(c, *m_top, pos);
                continue;
            }

            auto* curr = m_top - 1;
            while (curr->isDirty(c) && !curr->ctx.updates.requiresRefresh(c)) {
                --curr;
            }

            if (curr->ctx.updates.requiresRefresh(c)) {
                refresh(c, *m_top, pos);
                continue;
            }

            do {
                const auto& prev = *curr++;
                applyUpdates(c, curr->ctx.updates, prev.acc, *curr);
            } while (curr != m_top);
        }
    }

    i32 evaluateOnce(const Position& pos) {
        Accumulator acc{};
        acc.reset(pos);
        return forward(acc, pos.stm());
    }
} // namespace stoat::eval::nnue
