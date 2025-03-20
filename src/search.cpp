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

#include "core.h"
#include "eval/eval.h"
#include "history.h"
#include "movepick.h"
#include "protocol/handler.h"
#include "see.h"
#include "util/multi_array.h"

namespace stoat {
    namespace {
        constexpr f64 kWideningReportDelay = 1.5;

        constexpr auto kLmpTable = [] {
            std::array<i32, 32> result{};

            for (i32 depth = 0; depth < result.size(); ++depth) {
                result[depth] = 4 + 2 * depth * depth;
            }

            return result;
        }();

        // [depth][move index]
        const auto s_lmrTable = [] {
            constexpr f64 kBase = 0.2;
            constexpr f64 kDivisor = 3.0;

            util::MultiArray<i32, 256, 64> reductions{};

            for (i32 depth = 1; depth < 256; ++depth) {
                for (i32 moveNumber = 1; moveNumber < 64; ++moveNumber) {
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
        if (!m_ttable.finalize()) {
            m_ttable.clear();
        }

        for (auto& thread : m_threads) {
            thread.history.clear();
        }
    }

    void Searcher::ensureReady() {
        m_ttable.finalize();
    }

    void Searcher::setThreadCount(u32 threadCount) {
        assert(!isSearching());

        if (threadCount < 1) {
            threadCount = 1;
        }

        if (!m_threads.empty()) {
            stopThreads();
            m_quit.store(false);
        }

        m_threads.clear();
        m_threads.shrink_to_fit();
        m_threads.reserve(threadCount);

        m_resetBarrier.reset(threadCount + 1);
        m_idleBarrier.reset(threadCount + 1);

        m_searchEndBarrier.reset(threadCount);

        for (u32 threadId = 0; threadId < threadCount; ++threadId) {
            auto& thread = m_threads.emplace_back();

            thread.id = threadId;
            thread.thread = std::thread{[this, &thread] { runThread(thread); }};
        }
    }

    void Searcher::setTtSize(usize mib) {
        assert(!isSearching());
        m_ttable.resize(mib);
    }

    void Searcher::setCuteChessWorkaround(bool enabled) {
        assert(!isSearching());
        m_cuteChessWorkaround = enabled;
    }

    void Searcher::setLimiter(std::unique_ptr<limit::ISearchLimiter> limiter) {
        m_limiter = std::move(limiter);
    }

    void Searcher::startSearch(
        const Position& pos,
        std::span<const u64> keyHistory,
        util::Instant startTime,
        bool infinite,
        i32 maxDepth,
        std::unique_ptr<limit::ISearchLimiter> limiter
    ) {
        if (!limiter) {
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

        if (m_ttable.finalize()) {
            const auto initTime = initStart.elapsed();
            const auto ms = static_cast<u32>(initTime * 1000.0);
            protocol::currHandler().printInfoString(
                fmt::format("No newgame or isready before go, lost {} ms to TT initialization", ms)
            );
        }

        m_infinite = infinite;
        m_limiter = std::move(limiter);

        m_rootMoves = rootMoves;
        assert(!m_rootMoves.empty());

        for (auto& thread : m_threads) {
            thread.reset(pos, keyHistory);
            thread.maxDepth = maxDepth;

            thread.nnueState.reset(pos);
        }

        m_startTime = startTime;

        m_stop.store(false);
        m_runningThreads.store(m_threads.size());

        m_searching = true;

        m_idleBarrier.arriveAndWait();
    }

    void Searcher::stop() {
        m_stop.store(true, std::memory_order::relaxed);
        if (m_runningThreads.load() > 0) {
            std::unique_lock lock{m_stopMutex};
            m_stopSignal.wait(lock, [this] { return m_runningThreads.load() == 0; });
        }
    }

    ThreadData& Searcher::mainThread() {
        return m_threads[0];
    }

    void Searcher::runBenchSearch(BenchInfo& info, const Position& pos, i32 depth) {
        if (initRootMoves(m_rootMoves, pos) == RootStatus::kNoLegalMoves) {
            protocol::currHandler().printInfoString("no legal moves");
            return;
        }

        auto currLimiter = std::move(m_limiter);

        m_limiter = std::make_unique<limit::CompoundLimiter>();
        m_infinite = false;

        auto& thread = m_threads[0];

        thread.reset(pos, {});
        thread.maxDepth = depth;
        thread.nnueState.reset(pos);

        m_runningThreads.store(1);
        m_stop.store(false);

        m_startTime = util::Instant::now();

        runSearch(thread);

        info.time = m_startTime.elapsed();
        info.nodes = thread.loadNodes();

        m_limiter = std::move(currLimiter);
    }

    void Searcher::runDatagenSearch() {
        if (!m_limiter) {
            fmt::println(stderr, "Missing limiter");
            return;
        }

        if (m_threads.size() > 1) {
            fmt::println(stderr, "Too many datagen threads");
            return;
        }

        auto& thread = mainThread();
        thread.lastPv.reset();

        if (initRootMoves(m_rootMoves, thread.rootPos) == RootStatus::kNoLegalMoves) {
            return;
        }

        const bool wasInfinite = m_infinite;

        m_silent = true;
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

    Searcher::RootStatus Searcher::initRootMoves(movegen::MoveList& dst, const Position& pos) {
        dst.clear();
        generateLegal(dst, pos);
        return dst.empty() ? Searcher::RootStatus::kNoLegalMoves : Searcher::RootStatus::kGenerated;
    }

    void Searcher::runThread(ThreadData& thread) {
        while (true) {
            m_resetBarrier.arriveAndWait();
            m_idleBarrier.arriveAndWait();

            if (m_quit.load()) {
                return;
            }

            runSearch(thread);
        }
    }

    void Searcher::stopThreads() {
        m_quit.store(true);

        m_resetBarrier.arriveAndWait();
        m_idleBarrier.arriveAndWait();

        for (auto& thread : m_threads) {
            thread.thread.join();
        }
    }

    void Searcher::runSearch(ThreadData& thread) {
        assert(!m_rootMoves.empty());

        PvList rootPv{};

        thread.lastScore = kScoreNone;
        thread.lastPv.reset();

        for (i32 depth = 1;; ++depth) {
            thread.rootDepth = depth;
            thread.resetSeldepth();

            i32 window = 20;

            auto alpha = -kScoreInf;
            auto beta = kScoreInf;

            if (depth >= 3) {
                alpha = std::max(thread.lastScore - window, -kScoreInf);
                beta = std::min(thread.lastScore + window, kScoreInf);
            }

            Score score;

            while (true) {
                score = search<true, true>(thread, thread.rootPos, rootPv, depth, 0, alpha, beta);

                if (hasStopped()) {
                    break;
                }

                if (score > alpha && score < beta) {
                    break;
                }

                if (thread.isMainThread()) {
                    const auto time = m_startTime.elapsed();
                    if (time >= kWideningReportDelay) {
                        report(depth, score, alpha, beta, time, rootPv);
                    }
                }

                if (score <= alpha) {
                    alpha = std::max(score - window, -kScoreInf);
                } else { // score >= beta
                    beta = std::min(score + window, kScoreInf);
                }

                window += window;
            }

            if (hasStopped()) {
                break;
            }

            thread.depthCompleted = depth;

            thread.lastScore = score;
            thread.lastPv = rootPv;

            if (depth >= thread.maxDepth) {
                break;
            }

            if (thread.isMainThread()) {
                m_limiter->update(depth, thread.lastPv.moves[0]);

                if (m_limiter->stopSoft(thread.loadNodes())) {
                    break;
                }

                report(thread, m_startTime.elapsed());
            }
        }

        const auto waitForThreads = [&] {
            --m_runningThreads;
            m_stopSignal.notify_all();

            m_searchEndBarrier.arriveAndWait();
        };

        if (thread.isMainThread()) {
            const std::unique_lock lock{m_searchMutex};

            m_stop.store(true);
            waitForThreads();

            finalReport(m_startTime.elapsed());

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
        Score beta
    ) {
        assert(ply >= 0 && ply <= kMaxDepth);

        assert(kRootNode || ply > 0);
        assert(!kRootNode || ply == 0);

        assert(kPvNode || alpha == beta - 1);

        if (!kRootNode && thread.isMainThread() && thread.rootDepth > 1) {
            if (m_limiter->stopHard(thread.loadNodes())) {
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
            return pos.isInCheck() ? 0 : eval::staticEval(pos, thread.nnueState);
        }

        auto& curr = thread.stack[ply];
        const auto* parent = kRootNode ? nullptr : &thread.stack[ply - 1];

        tt::ProbedEntry ttEntry{};
        bool ttHit = false;

        if (!curr.excluded) {
            ttHit = m_ttable.probe(ttEntry, pos.key(), ply);

            if (!kPvNode && ttEntry.depth >= depth
                && (ttEntry.flag == tt::Flag::kExact                                   //
                    || ttEntry.flag == tt::Flag::kUpperBound && ttEntry.score <= alpha //
                    || ttEntry.flag == tt::Flag::kLowerBound && ttEntry.score >= beta))
            {
                return ttEntry.score;
            }

            if (depth >= 3 && !ttEntry.move) {
                --depth;
            }

            curr.staticEval = pos.isInCheck() ? kScoreNone : eval::staticEval(pos, thread.nnueState);
        }

        const bool improving = [&] {
            if (pos.isInCheck()) {
                return false;
            }
            if (ply > 1 && thread.stack[ply - 2].staticEval != kScoreNone) {
                return curr.staticEval > thread.stack[ply - 2].staticEval;
            }
            if (ply > 3 && thread.stack[ply - 4].staticEval != kScoreNone) {
                return curr.staticEval > thread.stack[ply - 4].staticEval;
            }
            return true;
        }();

        if (!kPvNode && !pos.isInCheck() && !curr.excluded) {
            if (depth <= 4 && curr.staticEval - 80 * (depth - improving) >= beta) {
                return curr.staticEval;
            }

            if (depth <= 4 && std::abs(alpha) < 2000 && curr.staticEval + 300 * depth <= alpha) {
                const auto score = qsearch(thread, pos, ply, alpha, alpha + 1);
                if (score <= alpha) {
                    return score;
                }
            }

            if (depth >= 4 && curr.staticEval >= beta && !parent->move.isNull()) {
                const auto r = 3 + depth / 5;

                const auto [newPos, guard] = thread.applyNullMove(ply, pos);
                const auto score = -search(thread, newPos, curr.pv, depth - r, ply + 1, -beta, -beta + 1);

                if (score >= beta) {
                    return score > kScoreWin ? beta : score;
                }
            }
        }

        auto bestMove = kNullMove;
        auto bestScore = -kScoreInf;

        auto ttFlag = tt::Flag::kUpperBound;

        auto generator = MoveGenerator::main(pos, ttEntry.move, thread.history);

        util::StaticVector<Move, 64> nonCapturesTried{};

        u32 legalMoves{};

        while (const auto move = generator.next()) {
            assert(pos.isPseudolegal(move));

            if (move == curr.excluded) {
                continue;
            }

            if constexpr (kRootNode) {
                if (!isLegalRootMove(move)) {
                    continue;
                }
                assert(pos.isLegal(move));
            } else if (!pos.isLegal(move)) {
                continue;
            }

            const auto baseLmr = s_lmrTable[depth][std::min<u32>(legalMoves, 63)];

            if (!kRootNode && bestScore > -kScoreWin && (!kPvNode || !thread.datagen)) {
                if (legalMoves >= kLmpTable[std::min<usize>(depth, kLmpTable.size())]) {
                    generator.skipNonCaptures();
                }

                const auto seeThreshold = pos.isCapture(move) ? -100 * depth * depth : -20 * depth * depth;
                if (!see::see(pos, move, seeThreshold)) {
                    continue;
                }

                if (depth <= 4 && !pos.isInCheck() && alpha < 2000 && !pos.isCapture(move)
                    && curr.staticEval + 150 + 100 * depth <= alpha)
                {
                    continue;
                }
            }

            if constexpr (kPvNode) {
                curr.pv.length = 0;
            }

            const auto prevNodes = thread.loadNodes();

            ++legalMoves;

            i32 extension{};

            if (!kRootNode && depth >= 7 && ply < thread.rootDepth * 2 && move == ttEntry.move && !curr.excluded
                && ttEntry.depth >= depth - 3 && ttEntry.flag != tt::Flag::kUpperBound)
            {
                const auto sBeta = std::max(-kScoreInf + 1, ttEntry.score - depth * 4 / 3);
                const auto sDepth = (depth - 1) / 2;

                curr.excluded = move;
                const auto score = search(thread, pos, curr.pv, sDepth, ply, sBeta - 1, sBeta);
                curr.excluded = kNullMove;

                if (score < sBeta) {
                    extension = 1;
                }
            }

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

                r += !kPvNode;
                r -= pos.isInCheck();
                r -= move.isDrop() && Square::chebyshev(move.to(), pos.kingSq(pos.stm().flip())) < 3;
                r += !improving;

                const auto reduced = std::min(std::max(newDepth - r, 1), newDepth - 1);
                score = -search(thread, newPos, curr.pv, reduced, ply + 1, -alpha - 1, -alpha);

                if (score > alpha && reduced < newDepth) {
                    score = -search(thread, newPos, curr.pv, newDepth, ply + 1, -alpha - 1, -alpha);
                }
            } else if (!kPvNode || legalMoves > 1) {
                score = -search(thread, newPos, curr.pv, newDepth, ply + 1, -alpha - 1, -alpha);
            }

            if (kPvNode && (legalMoves == 1 || score > alpha)) {
                score = -search<true>(thread, newPos, curr.pv, newDepth, ply + 1, -beta, -alpha);
            }

        skipSearch:
            if (hasStopped()) {
                return 0;
            }

            if (kRootNode && thread.isMainThread()) {
                m_limiter->addMoveNodes(move, thread.loadNodes() - prevNodes);
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

            if (move != bestMove && !pos.isCapture(move)) {
                nonCapturesTried.tryPush(move);
            }
        }

        if (legalMoves == 0) {
            assert(!kRootNode);
            return -kScoreMate + ply;
        }

        if (bestMove && !pos.isCapture(bestMove)) {
            const auto bonus = historyBonus(depth);
            thread.history.updateNonCaptureScore(bestMove, bonus);

            for (const auto prevNonCapture : nonCapturesTried) {
                thread.history.updateNonCaptureScore(prevNonCapture, -bonus);
            }
        }

        if (!curr.excluded) {
            m_ttable.put(pos.key(), bestScore, bestMove, depth, ply, ttFlag);
        }

        return bestScore;
    }

    template <bool kPvNode>
    Score Searcher::qsearch(ThreadData& thread, const Position& pos, i32 ply, Score alpha, Score beta) {
        assert(ply >= 0 && ply <= kMaxDepth);

        if (thread.isMainThread() && thread.rootDepth > 1) {
            if (m_limiter->stopHard(thread.loadNodes())) {
                m_stop.store(true, std::memory_order::relaxed);
                return 0;
            }
        }

        thread.incNodes();

        if constexpr (kPvNode) {
            thread.updateSeldepth(ply + 1);
        }

        if (ply >= kMaxDepth) {
            return pos.isInCheck() ? 0 : eval::staticEval(pos, thread.nnueState);
        }

        Score staticEval;

        if (pos.isInCheck()) {
            staticEval = -kScoreMate + ply;
        } else {
            staticEval = eval::staticEval(pos, thread.nnueState);

            if (staticEval >= beta) {
                return staticEval;
            }

            if (staticEval > alpha) {
                alpha = staticEval;
            }
        }

        auto bestScore = staticEval;

        auto generator = MoveGenerator::qsearch(pos, thread.history);

        while (const auto move = generator.next()) {
            assert(pos.isPseudolegal(move));

            if (!pos.isLegal(move)) {
                continue;
            }

            if (bestScore > -kScoreWin) {
                if (!see::see(pos, move, -100)) {
                    continue;
                }

                if (staticEval + 150 <= alpha && !see::see(pos, move, 1)) {
                    bestScore = std::max(bestScore, staticEval + 150);
                    continue;
                }
            }

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

    void Searcher::report(i32 depth, Score score, Score alpha, Score beta, f64 time, const PvList& pv) {
        if (m_silent) {
            return;
        }

        usize totalNodes = 0;
        i32 maxSeldepth = 0;

        for (const auto& thread : m_threads) {
            totalNodes += thread.loadNodes();
            maxSeldepth = std::max(maxSeldepth, thread.loadSeldepth());
        }

        auto bound = protocol::ScoreBound::kExact;

        if (score <= alpha) {
            bound = protocol::ScoreBound::kUpperBound;
        } else if (score >= beta) {
            bound = protocol::ScoreBound::kLowerBound;
        }

        score = std::clamp(score, alpha, beta);

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
            .depth = depth,
            .seldepth = maxSeldepth,
            .timeSec = time,
            .nodes = totalNodes,
            .score = displayScore,
            .scoreBound = bound,
            .pv = pv,
            .hashfull = m_ttable.fullPermille(),
        };

        protocol::currHandler().printSearchInfo(info);
    }

    void Searcher::report(const ThreadData& bestThread, f64 time) {
        report(bestThread.depthCompleted, bestThread.lastScore, -kScoreInf, kScoreInf, time, bestThread.lastPv);
    }

    void Searcher::finalReport(f64 time) {
        if (m_silent) {
            return;
        }

        const auto& bestThread = m_threads[0];

        report(bestThread, time);
        protocol::currHandler().printBestMove(bestThread.lastPv.moves[0]);
    }
} // namespace stoat
