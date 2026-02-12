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

#include "search.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "attacks/attacks.h"
#include "core.h"
#include "eval/eval.h"
#include "history.h"
#include "movepick.h"
#include "protocol/handler.h"
#include "see.h"
#include "stats.h"
#include "util/multi_array.h"

namespace stoat {
    namespace {
        constexpr f64 kWideningReportDelay = 1.5;

        constexpr usize kLmpTableSize = 32;

        constexpr auto kLmpTable = [] {
            util::MultiArray<i32, 2, kLmpTableSize> result{};

            for (i32 improving = 0; improving < 2; ++improving) {
                for (i32 depth = 0; depth < kLmpTableSize; ++depth) {
                    result[improving][depth] = (4 + 3 * depth * depth) / (2 - improving);
                }
            }

            return result;
        }();

        constexpr usize kLmrTableMoves = 64;

        // [depth][move index]
        const auto s_lmrTable = [] {
            constexpr f64 kBase = 0.5;
            constexpr f64 kDivisor = 2.5;

            util::MultiArray<i32, kMaxDepth, kLmrTableMoves> reductions{};

            for (i32 depth = 1; depth < kMaxDepth; ++depth) {
                for (i32 moveNumber = 1; moveNumber < kLmrTableMoves; ++moveNumber) {
                    const auto lnDepth = std::log(static_cast<f64>(depth));
                    const auto lnMoveNumber = std::log(static_cast<f64>(moveNumber));

                    reductions[depth][moveNumber] = static_cast<i32>(kBase + lnDepth * lnMoveNumber / kDivisor);
                }
            }

            return reductions;
        }();

        void generateLegal(movegen::MoveList& dst, const Position& pos) {
            movegen::MoveList generated{};
            movegen::generateAll(generated, pos);

            for (const auto move : generated) {
                if (pos.isLegal(move)) {
                    dst.push(move);
                }
            }
        }

        [[nodiscard]] constexpr Score drawScore(usize nodes) {
            return 2 - static_cast<Score>(nodes % 4);
        }

        [[nodiscard]] constexpr bool isDecisive(Score score) {
            return std::abs(score) > kScoreWin;
        }

        [[nodiscard]] bool isUnlikelyMove(const Position& pos, Move move) {
            if (move.isDrop() || move.isPromo()) {
                return false;
            }

            const auto pt = pos.pieceOn(move.from()).type();
            const auto promoArea = Bitboards::promoArea(pos.stm());

            if (pt != PieceTypes::kPawn && pt != PieceTypes::kLance && pt != PieceTypes::kBishop
                && pt != PieceTypes::kRook)
            {
                return false;
            }

            if (promoArea.getSquare(move.from())) {
                return true;
            }

            if (pt != PieceTypes::kLance && promoArea.getSquare(move.to())) {
                return true;
            }

            if (pt == PieceTypes::kLance && move.to().relative(pos.stm()).rank() == 7) {
                return true;
            }

            return false;
        }
    } // namespace

    Searcher::Searcher(usize ttSizeMb) :
            m_ttable{ttSizeMb} {
        setThreadCount(1);
    }

    Searcher::~Searcher() {
        stop();
        stopThreads();
    }

    void Searcher::newGame() {
        // Finalisation (init) clears the TT, so don't clear it twice
        if (!m_ttable.finalize(threadCount())) {
            m_ttable.clear(threadCount());
        }

        for (auto& thread : m_threadData) {
            thread->history.clear();
            thread->corrhist.clear();
        }
    }

    void Searcher::ensureReady() {
        m_ttable.finalize(threadCount());
    }

    void Searcher::setThreadCount(u32 threadCount) {
        assert(!isSearching());

        if (threadCount < 1) {
            threadCount = 1;
        }

        if (!m_threads.empty()) {
            stopThreads();
        }

        m_threads.clear();
        m_threads.shrink_to_fit();
        m_threads.reserve(threadCount);

        m_threadData.clear();
        m_threadData.resize(threadCount);
        m_threadData.shrink_to_fit();

        m_initBarrier.reset(threadCount + 1);

        m_resetBarrier.reset(threadCount + 1);
        m_idleBarrier.reset(threadCount + 1);

        m_searchEndBarrier.reset(threadCount);

        for (u32 threadId = 0; threadId < threadCount; ++threadId) {
            m_threads.emplace_back([this, threadId] { runThread(threadId); });
        }

        m_initBarrier.arriveAndWait();
    }

    void Searcher::setTtSize(usize mib) {
        assert(!isSearching());
        m_ttable.resize(mib);
    }

    void Searcher::setMultiPv(u32 multiPv) {
        assert(!isSearching());
        m_targetMultiPv = multiPv;
    }

    void Searcher::setMinimal(bool minimal) {
        assert(!isSearching());
        m_minimal = minimal;
    }

    void Searcher::setCuteChessWorkaround(bool enabled) {
        assert(!isSearching());
        m_cuteChessWorkaround = enabled;
    }

    void Searcher::setLimiter(limit::SearchLimiter limiter) {
        m_limiter = limiter;
    }

    void Searcher::startSearch(
        const Position& pos,
        std::span<const u64> keyHistory,
        util::Instant startTime,
        bool infinite,
        i32 maxDepth
    ) {
        if (!m_limiter) {
            fmt::println(stderr, "Missing limiter");
            return;
        }

        movegen::MoveList rootMoves{};
        const auto status = initRootMoves(rootMoves, pos);

        if (status == RootStatus::kNoLegalMoves) {
            protocol::currHandler().handleNoLegalMoves();
            return;
        }

        if (pos.isEnteringKingsWin() && protocol::currHandler().handleEnteringKingsWin()) {
            return;
        }

        m_resetBarrier.arriveAndWait();

        const std::unique_lock lock{m_searchMutex};

        const auto initStart = util::Instant::now();

        if (m_ttable.finalize(threadCount())) {
            const auto initTime = initStart.elapsed();
            const auto ms = static_cast<u32>(initTime * 1000.0);
            protocol::currHandler().printInfoString(
                fmt::format("No newgame or isready before go, lost {} ms to TT initialization", ms)
            );
        }

        m_infinite = infinite;

        m_rootMoveList = rootMoves;
        assert(!m_rootMoveList.empty());

        m_multiPv = std::min<u32>(m_targetMultiPv, m_rootMoveList.size());

        for (auto& thread : m_threadData) {
            thread->reset(pos, keyHistory);

            thread->limiter = m_limiter;
            thread->maxDepth = maxDepth;
            thread->stoppedSoft = false;

            thread->nnueState.reset(pos);
        }

        m_startTime = startTime;

        m_stop.store(false);

        m_runningThreads.store(m_threads.size());
        m_softStoppedThreads.store(0);

        m_searching = true;

        m_idleBarrier.arriveAndWait();
    }

    void Searcher::stop() {
        m_stop.store(true, std::memory_order::relaxed);

        std::unique_lock lock{m_stopMutex};
        if (m_runningThreads.load() > 0) {
            m_stopSignal.wait(lock, [this] { return m_runningThreads.load() == 0; });
        }

        m_stop.store(false);
    }

    ThreadData& Searcher::take() {
        stopThreads();

        m_resetBarrier.reset(1);
        m_idleBarrier.reset(1);

        m_threads.clear();
        m_threads.shrink_to_fit();

        m_threadData.resize(1);
        m_threadData.shrink_to_fit();

        m_threadData[0] = std::make_unique<ThreadData>();

        return *m_threadData[0];
    }

    void Searcher::runBenchSearch(BenchInfo& info) {
        auto& thread = *m_threadData[0];

        if (initRootMoves(m_rootMoveList, thread.rootPos) == RootStatus::kNoLegalMoves) {
            protocol::currHandler().printInfoString("no legal moves");
            return;
        }

        const auto currLimiter = m_limiter;

        m_limiter = limit::SearchLimiter{util::Instant::now()};

        m_multiPv = 1;
        m_infinite = false;

        thread.nnueState.reset(thread.rootPos);

        m_runningThreads.store(1);
        m_stop.store(false);

        m_startTime = util::Instant::now();

        runSearch(thread);

        info.time = m_startTime.elapsed();
        info.nodes = thread.loadNodes();

        m_limiter = currLimiter;
    }

    void Searcher::runDatagenSearch() {
        if (!m_limiter) {
            fmt::println(stderr, "Missing limiter");
            return;
        }

        auto& thread = *m_threadData[0];

        if (initRootMoves(m_rootMoveList, thread.rootPos) == RootStatus::kNoLegalMoves) {
            return;
        }

        const bool wasInfinite = m_infinite;

        m_silent = true;

        m_multiPv = 1;
        m_infinite = false;

        m_stop.store(false);
        ++m_runningThreads;

        runSearch(thread);

        m_silent = false;
        m_infinite = wasInfinite;
    }

    bool Searcher::isSearching() const {
        const std::unique_lock lock{m_searchMutex};
        return m_searching;
    }

    u32 Searcher::threadCount() const {
        return m_threads.size();
    }

    Searcher::RootStatus Searcher::initRootMoves(movegen::MoveList& dst, const Position& pos) {
        dst.clear();
        generateLegal(dst, pos);
        return dst.empty() ? Searcher::RootStatus::kNoLegalMoves : Searcher::RootStatus::kGenerated;
    }

    void Searcher::runThread(u32 id) {
        m_threadData[id] = std::make_unique<ThreadData>();

        auto& thread = *m_threadData[id];
        thread.id = id;

        m_initBarrier.arriveAndWait();

        while (true) {
            m_resetBarrier.arriveAndWait();
            m_idleBarrier.arriveAndWait();

            if (m_quit.load()) {
                return;
            }

            runSearch(thread);
        }
    }

    void Searcher::signalThreadSoftStopped() {
        const auto stopped = ++m_softStoppedThreads;
        const auto voteThreshold = (m_threadData.size() + 1) / 2;
        if (stopped >= voteThreshold) {
            m_stop.store(true);
        }
    }

    void Searcher::stopThreads() {
        stop();

        m_quit.store(true);

        m_resetBarrier.arriveAndWait();
        m_idleBarrier.arriveAndWait();

        for (auto& thread : m_threads) {
            thread.join();
        }

        m_quit.store(false);
    }

    void Searcher::runSearch(ThreadData& thread) {
        assert(!m_rootMoveList.empty());

        thread.rootMoves.clear();
        thread.rootMoves.reserve(m_rootMoveList.size());

        for (const auto move : m_rootMoveList) {
            auto& rootMove = thread.rootMoves.emplace_back();

            rootMove.pv.moves[0] = move;
            rootMove.pv.length = 1;
        }

        PvList rootPv{};

        for (i32 depth = 1;; ++depth) {
            thread.rootDepth = depth;

            for (auto& move : thread.rootMoves) {
                move.previousScore = move.score;
            }

            for (thread.pvIdx = 0; thread.pvIdx < m_multiPv; ++thread.pvIdx) {
                thread.resetSeldepth();

                i32 window = 20;

                auto alpha = -kScoreInf;
                auto beta = kScoreInf;

                if (depth >= 3) {
                    const auto lastScore = thread.rootMoves[thread.pvIdx].windowScore;
                    alpha = std::max(lastScore - window, -kScoreInf);
                    beta = std::min(lastScore + window, kScoreInf);
                }

                Score score;

                i32 reduction{};

                while (true) {
                    const auto rootDepth = std::max(depth - reduction, 1);

                    score = search<true, true>(thread, thread.rootPos, rootPv, rootDepth, 0, alpha, beta, false);

                    std::stable_sort(
                        thread.rootMoves.begin() + thread.pvIdx,
                        thread.rootMoves.end(),
                        [](const RootMove& a, const RootMove& b) { return a.score > b.score; }
                    );

                    if (hasStopped()) {
                        break;
                    }

                    if (score > alpha && score < beta) {
                        break;
                    }

                    if (thread.isMainThread() && !m_minimal && m_targetMultiPv == 1) {
                        const auto time = m_startTime.elapsed();
                        if (time >= kWideningReportDelay) {
                            reportSingle(thread, thread.pvIdx, depth, time);
                        }
                    }

                    if (score <= alpha) {
                        reduction = 0;

                        beta = (alpha + beta) / 2;
                        alpha = std::max(score - window, -kScoreInf);
                    } else { // score >= beta
                        reduction = std::min(reduction + 1, 3);
                        beta = std::min(score + window, kScoreInf);
                    }

                    window += window;
                }

                std::ranges::stable_sort(thread.rootMoves, [](const RootMove& a, const RootMove& b) {
                    return a.score > b.score;
                });

                if (hasStopped()) {
                    break;
                }
            }

            if (hasStopped()) {
                break;
            }

            thread.depthCompleted = depth;

            if (depth >= thread.maxDepth) {
                break;
            }

            const auto nodes = thread.loadNodes();

            thread.limiter->update(depth, nodes, thread.pvMove());

            if (!thread.stoppedSoft && thread.limiter->stopSoft(nodes)) {
                thread.stoppedSoft = true;
                signalThreadSoftStopped();
                if (hasStopped()) {
                    break;
                }
            }

            if (thread.isMainThread() && !m_minimal) {
                report(thread, depth, m_startTime.elapsed());
            }
        }

        const auto waitForThreads = [&] {
            {
                const std::unique_lock lock{m_stopMutex};
                --m_runningThreads;
                m_stopSignal.notify_all();
            }

            m_searchEndBarrier.arriveAndWait();
        };

        if (thread.isMainThread()) {
            const std::unique_lock lock{m_searchMutex};

            m_stop.store(true);
            waitForThreads();

            finalReport(m_startTime.elapsed());

            m_ttable.age();
            stats::print();

            m_searching = false;
        } else {
            waitForThreads();
        }
    }

    template <bool kPvNode, bool kRootNode>
    Score Searcher::search(
        ThreadData& thread,
        const Position& pos,
        PvList& pv,
        i32 depth,
        i32 ply,
        Score alpha,
        Score beta,
        bool expectedCutnode
    ) {
        assert(ply >= 0 && ply <= kMaxDepth);

        assert(kRootNode || ply > 0);
        assert(!kRootNode || ply == 0);

        assert(kPvNode || alpha == beta - 1);
        assert(!kPvNode || !expectedCutnode);

        if (hasStopped()) {
            return 0;
        }

        if (!kRootNode && thread.isMainThread() && thread.rootDepth > 1) {
            if (thread.limiter->stopHard(thread.loadNodes())) {
                m_stop.store(true, std::memory_order::relaxed);
                return 0;
            }
        }

        if constexpr (!kRootNode) {
            alpha = std::max(alpha, -kScoreMate + ply);
            beta = std::min(beta, kScoreMate - ply - 1);

            if (alpha >= beta) {
                return alpha;
            }
        }

        if (depth <= 0) {
            return qsearch<kPvNode>(thread, pos, ply, alpha, beta);
        }

        thread.incNodes();

        if constexpr (kPvNode) {
            thread.updateSeldepth(ply + 1);
        }

        if (ply >= kMaxDepth) {
            return pos.isInCheck() ? 0 : eval::adjustedEval(pos, thread.nnueState, thread.corrhist, ply);
        }

        auto& curr = thread.stack[ply];
        const auto* parent = kRootNode ? nullptr : &thread.stack[ply - 1];

        tt::ProbedEntry ttEntry{};
        bool ttHit = false;

        if (!curr.excluded) {
            ttHit = m_ttable.probe(ttEntry, pos.key(), ply);

            if (!kPvNode && ttEntry.depth >= depth
                && (ttEntry.flag == tt::Flag::kExact                                     //
                    || (ttEntry.flag == tt::Flag::kUpperBound && ttEntry.score <= alpha) //
                    || (ttEntry.flag == tt::Flag::kLowerBound && ttEntry.score >= beta)))
            {
                return ttEntry.score;
            }

            if (depth >= 3 && !ttEntry.move) {
                --depth;
            }
        }

        const bool ttPv = ttEntry.pv || kPvNode;

        Score rawEval{};

        if (!curr.excluded) {
            if (pos.isInCheck()) {
                rawEval = kScoreNone;
                curr.staticEval = kScoreNone;
            } else {
                if (ttHit && ttEntry.staticEval != kScoreNone) {
                    rawEval = ttEntry.staticEval;
                } else {
                    rawEval = eval::staticEval(pos, thread.nnueState);
                    if (!ttHit) {
                        m_ttable.putStaticEval(pos.key(), rawEval, ttPv);
                    }
                }

                curr.staticEval = eval::adjustEval(rawEval, pos, thread.corrhist, ply);
            }
        }

        const auto complexity = [&] {
            if (ttEntry.flag == tt::Flag::kExact                                               //
                || (ttEntry.flag == tt::Flag::kUpperBound && ttEntry.score <= curr.staticEval) //
                || (ttEntry.flag == tt::Flag::kLowerBound && ttEntry.score >= curr.staticEval))
            {
                return std::abs(curr.staticEval - ttEntry.score);
            }
            return 0;
        }();

        const auto ttMove =
            (kRootNode && thread.rootDepth > 1) ? thread.rootMoves[thread.pvIdx].pv.moves[0] : ttEntry.move;

        const bool improving = [&] {
            if (pos.isInCheck()) {
                return false;
            }
            if (ply > 1 && thread.stack[ply - 2].staticEval != kScoreNone) {
                return curr.staticEval > thread.stack[ply - 2].staticEval + 50 * expectedCutnode;
            }
            if (ply > 3 && thread.stack[ply - 4].staticEval != kScoreNone) {
                return curr.staticEval > thread.stack[ply - 4].staticEval + 100 * expectedCutnode;
            }
            return false;
        }();

        if (!ttPv && !pos.isInCheck() && !curr.excluded && complexity <= 20) {
            if (parent && depth >= 2 && parent->reduction >= 1 && curr.staticEval + parent->staticEval >= 200) {
                depth--;
            }

            if (parent && parent->reduction >= 3 && curr.staticEval + parent->staticEval <= 0) {
                depth++;
            }

            if (depth <= 10 && curr.staticEval - 80 * (depth - improving) >= beta) {
                return curr.staticEval;
            }

            if (depth <= 4 && std::abs(alpha) < 2000 && curr.staticEval + 300 * depth <= alpha) {
                const auto score = qsearch(thread, pos, ply, alpha, alpha + 1);
                if (score <= alpha) {
                    return score;
                }
            }

            if (depth >= 4 && curr.staticEval >= beta + 70 && !parent->move.isNull()) {
                const auto r = 3 + depth / 5;

                const auto [newPos, guard] = thread.applyNullMove(ply, pos);
                const auto score =
                    -search(thread, newPos, curr.pv, depth - r, ply + 1, -beta, -beta + 1, !expectedCutnode);

                if (score >= beta) {
                    return score > kScoreWin ? beta : score;
                }
            }
        }

        if (!ttPv && ttMove && !pos.isInCheck() && !curr.excluded && std::abs(ttEntry.score) < kScoreMaxMate
            && ttEntry.score >= beta + 300 && ttEntry.depth >= depth - 4 && pos.isPseudolegal(ttEntry.move))
        {
            return ttEntry.score;
        }

        auto bestMove = kNullMove;
        auto bestScore = -kScoreInf;

        auto ttFlag = tt::Flag::kUpperBound;

        auto generator = MoveGenerator::main(pos, ttMove, thread.history, thread.conthist, ply);

        util::StaticVector<Move, 64> capturesTried{};
        util::StaticVector<Move, 64> nonCapturesTried{};

        u32 legalMoves{};

        while (const auto move = generator.next()) {
            assert(pos.isPseudolegal(move));

            if (move == curr.excluded) {
                continue;
            }

            if constexpr (kRootNode) {
                if (!thread.isLegalRootMove(move)) {
                    continue;
                }
                assert(pos.isLegal(move));
            } else if (!pos.isLegal(move)) {
                continue;
            }

            if (isUnlikelyMove(pos, move) && curr.staticEval - 500 <= alpha) {
                continue;
            }

            const auto baseLmr = s_lmrTable[depth][std::min<u32>(legalMoves, kLmrTableMoves - 1)];
            const auto history = pos.isCapture(move) ? 0 : thread.history.mainNonCaptureScore(pos, move);

            if (!kRootNode && bestScore > -kScoreWin && (!kPvNode || !thread.datagen)) {
                if (legalMoves >= kLmpTable[improving][std::min<usize>(depth, kLmpTableSize - 1)]) {
                    generator.skipNonCaptures();
                }

                if (depth <= 8 && !pos.isInCheck() && alpha < 2000 && !pos.isCapture(move)
                    && curr.staticEval + 150 + 100 * depth + history / 128 <= alpha)
                {
                    continue;
                }

                const auto seeThreshold = pos.isCapture(move) ? -77 * depth * depth : -15 * depth * depth;
                if (!see::see(pos, move, seeThreshold)) {
                    continue;
                }
            }

            if constexpr (kPvNode) {
                curr.pv.length = 0;
            }

            const auto prevNodes = thread.loadNodes();

            ++legalMoves;

            i32 extension{};

            if (!kRootNode && ply < thread.rootDepth * 2 && move == ttMove && !curr.excluded) {
                if (depth >= 6 && ttEntry.depth >= depth - 3 && ttEntry.flag != tt::Flag::kUpperBound
                    && !isDecisive(ttEntry.score))
                {
                    const auto sBeta = ttEntry.score - depth * 4 / 3;
                    const auto sDepth = (depth - 1) / 2;

                    curr.excluded = move;
                    const auto score = search(thread, pos, curr.pv, sDepth, ply, sBeta - 1, sBeta, expectedCutnode);
                    curr.excluded = kNullMove;

                    if (score < sBeta) {
                        if (!kPvNode && !pos.isCapture(move) && score < sBeta - 25) {
                            extension = 2;
                        } else {
                            extension = 1;
                        }
                    } else if (sBeta >= beta) {
                        return sBeta;
                    } else if (ttEntry.score >= beta) {
                        extension = -1;
                    } else if (expectedCutnode) {
                        extension = -1;
                    }
                } else if (depth <= 7 && !pos.isInCheck() && curr.staticEval <= alpha - 26
                           && ttEntry.flag == tt::Flag::kLowerBound)
                {
                    extension = 1;
                }
            }

            m_ttable.prefetch(pos.keyAfter(move));

            const auto [newPos, guard] = thread.applyMove(ply, pos, move);
            const auto sennichite = newPos.testSennichite(m_cuteChessWorkaround, thread.keyHistory);

            const bool givesCheck = newPos.isInCheck();

            auto newDepth = depth - 1;

            Score score;

            if (sennichite == SennichiteStatus::kWin) {
                // illegal perpetual
                --legalMoves;
                continue;
            } else if (sennichite == SennichiteStatus::kDraw) {
                score = drawScore(thread.loadNodes());
                goto skipSearch;
            } else if (pos.isEnteringKingsWin()) {
                score = kScoreMate - ply - 1;
                goto skipSearch;
            }

            if (extension == 0 && givesCheck) {
                extension = 1;
            }

            newDepth += extension;

            if (depth >= 2 && legalMoves >= 3 + 2 * kRootNode && !givesCheck
                && generator.stage() >= MovegenStage::kNonCaptures)
            {
                auto r = baseLmr;
                const auto dist = Square::chebyshev(move.to(), pos.kingSq(pos.stm().flip()));

                r += !ttPv;
                r += !improving;
                r -= history / 8192;
                r += expectedCutnode * 3;

                if (pos.isInCheck()) {
                    r -= 1 + (dist == 1);
                }

                if (pos.isCapture(move)) {
                    r -= 1 + (see::pieceValue(pos.pieceOn(move.to()).type()) + 150) / 250;
                }

                if (move.isDrop()) {
                    r -= dist < 3 && !pos.attackersTo(move.to(), pos.stm()).empty();
                    r -= (attacks::pieceAttacks(move.dropPiece(), move.to(), pos.stm(), pos.occupancy())
                          & pos.colorBb(pos.stm().flip()))
                             .popcount();
                }

                const auto reduced = std::min(std::max(newDepth - r, 1), newDepth - 1) + kPvNode;
                curr.reduction = newDepth - reduced;
                score = -search(thread, newPos, curr.pv, reduced, ply + 1, -alpha - 1, -alpha, true);
                curr.reduction = 0;

                if (score > alpha && reduced < newDepth) {
                    score = -search(thread, newPos, curr.pv, newDepth, ply + 1, -alpha - 1, -alpha, !expectedCutnode);
                    if (!pos.isCapture(move) && score >= beta) {
                        const auto bonus = historyBonus(newDepth);
                        thread.history.updateNonCaptureConthistScore(thread.conthist, ply, pos, move, bonus);
                    }
                }
            } else if (!kPvNode || legalMoves > 1) {
                score = -search(thread, newPos, curr.pv, newDepth, ply + 1, -alpha - 1, -alpha, !expectedCutnode);
            }

            if (kPvNode && (legalMoves == 1 || score > alpha)) {
                score = -search<true>(thread, newPos, curr.pv, newDepth, ply + 1, -beta, -alpha, false);
            }

        skipSearch:
            if (hasStopped()) {
                return 0;
            }

            if (kRootNode) {
                auto* rootMove = thread.findRootMove(move);

                if (!rootMove) {
                    fmt::println(stderr, "Failed to find root move for {}", move);
                    std::terminate();
                }

                rootMove->windowScore = score;
                rootMove->nodes += thread.loadNodes() - prevNodes;

                if (legalMoves == 1 || score > alpha) {
                    rootMove->seldepth = thread.loadSeldepth();

                    rootMove->displayScore = score;
                    rootMove->score = score;

                    rootMove->upperbound = false;
                    rootMove->lowerbound = false;

                    if (score <= alpha) {
                        rootMove->displayScore = alpha;
                        rootMove->upperbound = true;
                    } else if (score >= beta) {
                        rootMove->displayScore = beta;
                        rootMove->lowerbound = true;
                    }

                    rootMove->pv.update(move, curr.pv);
                } else {
                    rootMove->score = -kScoreInf;
                }
            }

            if (score > bestScore) {
                bestScore = score;
            }

            if (score > alpha) {
                alpha = score;
                bestMove = move;

                if constexpr (kPvNode) {
                    assert(curr.pv.length + 1 <= kMaxDepth);
                    pv.update(move, curr.pv);
                }

                ttFlag = tt::Flag::kExact;
            }

            if (score >= beta) {
                ttFlag = tt::Flag::kLowerBound;
                break;
            }

            if (move != bestMove) {
                if (pos.isCapture(move)) {
                    capturesTried.tryPush(move);
                } else {
                    nonCapturesTried.tryPush(move);
                }
            }
        }

        if (legalMoves == 0) {
            assert(!kRootNode);
            return -kScoreMate + ply;
        }

        if (bestMove) {
            const auto historyDepth = depth + (!pos.isInCheck() && curr.staticEval <= bestScore);
            const auto bonus = historyBonus(historyDepth);

            if (!pos.isCapture(bestMove)) {
                thread.history.updateNonCaptureScore(thread.conthist, ply, pos, bestMove, bonus);

                for (const auto prevNonCapture : nonCapturesTried) {
                    thread.history.updateNonCaptureScore(thread.conthist, ply, pos, prevNonCapture, -bonus);
                }
            } else {
                const auto captured = pos.pieceOn(bestMove.to()).type();
                thread.history.updateCaptureScore(bestMove, captured, bonus);
            }

            for (const auto prevCapture : capturesTried) {
                const auto captured = pos.pieceOn(prevCapture.to()).type();
                thread.history.updateCaptureScore(prevCapture, captured, -bonus);
            }
        }

        if (bestScore >= beta && !isDecisive(bestScore) && !isDecisive(beta)) {
            bestScore = (bestScore * depth + beta) / (depth + 1);
        }

        if (!curr.excluded) {
            if (!pos.isInCheck() && (bestMove.isNull() || !pos.isCapture(bestMove))
                && (ttFlag == tt::Flag::kExact                                          //
                    || (ttFlag == tt::Flag::kUpperBound && bestScore < curr.staticEval) //
                    || (ttFlag == tt::Flag::kLowerBound && bestScore > curr.staticEval)))
            {
                thread.corrhist.update(pos, depth, bestScore, curr.staticEval, complexity);
            }

            if (!kRootNode || thread.pvIdx == 0) {
                m_ttable.put(pos.key(), bestScore, rawEval, bestMove, depth, ply, ttFlag, ttPv);
            }
        }

        return bestScore;
    }

    template <bool kPvNode>
    Score Searcher::qsearch(ThreadData& thread, const Position& pos, i32 ply, Score alpha, Score beta) {
        assert(ply >= 0 && ply <= kMaxDepth);

        if (hasStopped()) {
            return 0;
        }

        if (thread.isMainThread() && thread.rootDepth > 1) {
            if (thread.limiter->stopHard(thread.loadNodes())) {
                m_stop.store(true, std::memory_order::relaxed);
                return 0;
            }
        }

        thread.incNodes();

        if constexpr (kPvNode) {
            thread.updateSeldepth(ply + 1);
        }

        if (ply >= kMaxDepth) {
            return pos.isInCheck() ? 0 : eval::adjustedEval(pos, thread.nnueState, thread.corrhist, ply);
        }

        Score rawEval, staticEval;

        if (pos.isInCheck()) {
            rawEval = kScoreNone;
            staticEval = -kScoreMate + ply;
        } else {
            rawEval = eval::staticEval(pos, thread.nnueState);
            staticEval = eval::adjustEval(rawEval, pos, thread.corrhist, ply);

            if (staticEval >= beta) {
                return staticEval;
            }

            if (staticEval > alpha) {
                alpha = staticEval;
            }
        }

        auto bestScore = staticEval;

        auto generator = MoveGenerator::qsearch(pos, thread.history, thread.conthist, ply);

        u32 legalMoves{};

        while (const auto move = generator.next()) {
            assert(pos.isPseudolegal(move));

            if (!pos.isLegal(move) || isUnlikelyMove(pos, move)) {
                continue;
            }

            if (bestScore > -kScoreWin) {
                if (!see::see(pos, move, -77)) {
                    continue;
                }

                if (staticEval + 150 <= alpha && !see::see(pos, move, 1)) {
                    bestScore = std::max(bestScore, staticEval + 150);
                    continue;
                }

                if (legalMoves >= 2) {
                    break;
                }
            }

            ++legalMoves;

            const auto [newPos, guard] = thread.applyMove(ply, pos, move);
            const auto sennichite = newPos.testSennichite(m_cuteChessWorkaround, thread.keyHistory);

            Score score;

            if (sennichite == SennichiteStatus::kWin) {
                // illegal perpetual
                continue;
            } else if (sennichite == SennichiteStatus::kDraw) {
                score = drawScore(thread.loadNodes());
            } else {
                score = -qsearch<kPvNode>(thread, newPos, ply + 1, -beta, -alpha);
            }

            if (hasStopped()) {
                return 0;
            }

            if (score > -kScoreWin) {
                generator.skipNonCaptures();
            }

            if (score > bestScore) {
                bestScore = score;
            }

            if (score > alpha) {
                alpha = score;
            }

            if (score >= beta) {
                break;
            }
        }

        return bestScore;
    }

    void Searcher::reportSingle(const ThreadData& bestThread, u32 pvIdx, i32 depth, f64 time) const {
        if (m_silent) {
            return;
        }

        const auto& move = bestThread.rootMoves[pvIdx];

        auto score = move.displayScore;

        if (move.score == -kScoreInf) {
            score = move.previousScore;
            depth = std::max(1, depth - 1);
        }

        usize totalNodes = 0;

        for (const auto& thread : m_threadData) {
            totalNodes += thread->loadNodes();
        }

        auto bound = protocol::ScoreBound::kExact;

        if (move.upperbound) {
            bound = protocol::ScoreBound::kUpperBound;
        } else if (move.lowerbound) {
            bound = protocol::ScoreBound::kLowerBound;
        }

        protocol::DisplayScore displayScore;

        if (std::abs(score) >= kScoreMaxMate) {
            if (score > 0) {
                displayScore = protocol::MateDisplayScore{kScoreMate - score};
            } else {
                displayScore = protocol::MateDisplayScore{-(kScoreMate + score)};
            }
        } else {
            // clamp draw scores to 0
            if (std::abs(score) <= 2) {
                score = 0;
            }

            displayScore = protocol::CpDisplayScore{score};
        }

        const protocol::SearchInfo info = {
            .pvIdx = pvIdx,
            .multiPv = m_multiPv,
            .depth = depth,
            .seldepth = move.seldepth,
            .timeSec = time,
            .nodes = totalNodes,
            .score = displayScore,
            .scoreBound = bound,
            .pv = move.pv,
            .hashfull = m_ttable.fullPermille(),
        };

        protocol::currHandler().printSearchInfo(info);
    }

    void Searcher::report(const ThreadData& bestThread, i32 depth, f64 time) const {
        for (u32 pvIdx = 0; pvIdx < m_multiPv; ++pvIdx) {
            reportSingle(bestThread, pvIdx, depth, time);
        }
    }

    const ThreadData& Searcher::selectThread() const {
        if (m_threadData.size() == 1) {
            return *m_threadData[0];
        }

        // Thread selection logic ported from SF

        auto lowestRootScore = kScoreInf;

        for (const auto& thread : m_threadData) {
            const auto score = thread->pvMove().score;
            if (score != -kScoreInf && score < lowestRootScore) {
                lowestRootScore = score;
            }
        }

        const auto threadWeight = [&](const ThreadData& thread) {
            return (thread.pvMove().score - lowestRootScore + 10) * thread.depthCompleted;
        };

        std::unordered_map<u16, i32> moveVotes{};

        const auto bestMoveVotes = [&](const ThreadData& thread) -> i32& {
            return moveVotes[thread.pvMove().pv.moves[0].raw()];
        };

        for (const auto& thread : m_threadData) {
            if (thread->pvMove().score != -kScoreInf) {
                bestMoveVotes(*thread) += threadWeight(*thread);
            }
        }

        const auto* bestThread = m_threadData[0].get();

        auto bestRootScore = bestThread->pvMove().score;
        auto bestVotes = bestMoveVotes(*bestThread);

        const auto acceptCandidateThread = [&](const ThreadData* thread) {
            bestThread = thread;
            bestRootScore = thread->pvMove().score;
            bestVotes = bestMoveVotes(*thread);
        };

        for (usize threadIdx = 1; threadIdx < m_threadData.size(); ++threadIdx) {
            const auto* thread = m_threadData[threadIdx].get();

            const auto rootScore = thread->pvMove().score;

            if (rootScore == -kScoreInf) {
                continue;
            }

            if (bestRootScore == -kScoreInf) {
                acceptCandidateThread(thread);
                continue;
            }

            // if the best thread has a proven win, only take a better win
            if (bestRootScore > kScoreWin) {
                if (rootScore > bestRootScore) {
                    acceptCandidateThread(thread);
                }
                continue;
            }

            // if the best thread has a proven loss, take the candidate if it has a shorter one
            if (bestRootScore < -kScoreWin) {
                if (rootScore < -kScoreWin && rootScore < bestRootScore) {
                    acceptCandidateThread(thread);
                }
                continue;
            }

            // if the best thread has any proven decisive score, take it
            if (std::abs(rootScore) > kScoreWin) {
                acceptCandidateThread(thread);
                continue;
            }

            const auto votes = bestMoveVotes(*thread);

            // take the move with the most votes
            if (votes > bestVotes) {
                acceptCandidateThread(thread);
                continue;
            }

            // given moves with equal votes, take the highest weighted thread,
            //   but avoid truncated PVs from failhighs/lows
            if (votes == bestVotes
                && (threadWeight(*thread) * (thread->pvMove().pv.length > 2))
                       > (threadWeight(*bestThread) * (bestThread->pvMove().pv.length > 2)))
            {
                acceptCandidateThread(thread);
                continue;
            }
        }

        if (!m_minimal) {
            fmt::println("info string Selected thread {}", bestThread->id);
        }

        return *bestThread;
    }

    void Searcher::finalReport(f64 time) const {
        if (m_silent) {
            return;
        }

        const auto& bestThread = selectThread();

        report(bestThread, bestThread.depthCompleted, time);
        protocol::currHandler().printBestMove(bestThread.pvMove().pv.moves[0]);
    }
} // namespace stoat
