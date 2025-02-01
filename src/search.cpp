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

#include "eval/eval.h"
#include "movepick.h"
#include "protocol/handler.h"
#include "util/multi_array.h"

namespace stoat {
    namespace {
        // [depth][move index]
        auto s_lmrTable = [] {
            constexpr f64 kBase = 0.2;
            constexpr f64 kDivisor = 3.5;

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
        setThreads(1);
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
    }

    void Searcher::ensureReady() {
        m_ttable.finalize();
    }

    void Searcher::setThreads(u32 threadCount) {
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

    void Searcher::startSearch(
        const Position& pos,
        std::span<const u64> keyHistory,
        util::Instant startTime,
        bool infinite,
        i32 maxDepth,
        std::unique_ptr<limit::ISearchLimiter> limiter
    ) {
        if (!limiter) {
            std::cerr << "Missing limiter" << std::endl;
            return;
        }

        m_resetBarrier.arriveAndWait();

        const std::unique_lock lock{m_searchMutex};

        const auto initStart = util::Instant::now();

        if (m_ttable.finalize()) {
            const auto initTime = initStart.elapsed();
            const auto ms = static_cast<u32>(initTime * 1000.0);
            protocol::currHandler().printInfoString(
                std::cout,
                "No newgame or isready before go, lost " + std::to_string(ms) + " ms to TT initialization"
            );
        }

        m_infinite = infinite;
        m_limiter = std::move(limiter);

        const auto status = initRootMoves(pos);

        if (status == RootStatus::kNoLegalMoves) {
            const PvList pv{};
            const protocol::SearchInfo info = {
                .depth = 1,
                .nodes = 0,
                .score = protocol::MateDisplayScore{0},
                .pv = pv,
            };

            protocol::currHandler().printInfoString(std::cout, "no legal moves");
            protocol::currHandler().printSearchInfo(std::cout, info);

            return;
        }

        assert(!m_rootMoves.empty());

        for (auto& thread : m_threads) {
            thread.reset(pos, keyHistory);
            thread.maxDepth = maxDepth;
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

    void Searcher::runBenchSearch(BenchInfo& info, const Position& pos, i32 depth) {
        if (initRootMoves(pos) == RootStatus::kNoLegalMoves) {
            protocol::currHandler().printInfoString(std::cout, "no legal moves");
            return;
        }

        m_limiter = std::make_unique<limit::CompoundLimiter>();
        m_infinite = false;

        auto& thread = m_threads[0];

        thread.reset(pos, {});
        thread.maxDepth = depth;

        m_runningThreads.store(1);
        m_stop.store(false);

        m_startTime = util::Instant::now();

        runSearch(thread);

        info.time = m_startTime.elapsed();
        info.nodes = thread.loadNodes();

        m_limiter = nullptr;
    }

    bool Searcher::isSearching() const {
        const std::unique_lock lock{m_searchMutex};
        return m_searching;
    }

    Searcher::RootStatus Searcher::initRootMoves(const Position& pos) {
        m_rootMoves.clear();
        generateLegal(m_rootMoves, pos);
        return m_rootMoves.empty() ? Searcher::RootStatus::kNoLegalMoves : Searcher::RootStatus::kGenerated;
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

            const auto score = search<true, true>(thread, thread.rootPos, rootPv, depth, 0, -kScoreInf, kScoreInf);

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

            m_limiter = nullptr;
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

        if (depth <= 0) {
            return qsearch<kPvNode>(thread, pos, ply, alpha, beta);
        }

        thread.incNodes();

        if constexpr (kPvNode) {
            thread.updateSeldepth(ply + 1);
        }

        if (ply >= kMaxDepth) {
            return pos.isInCheck() ? 0 : eval::staticEval(pos);
        }

        auto& curr = thread.stack[ply];

        tt::ProbedEntry ttEntry{};
        const bool ttHit = m_ttable.probe(ttEntry, pos.key(), ply);

        if (!kPvNode && ttEntry.depth >= depth
            && (ttEntry.flag == tt::Flag::kExact                                   //
                || ttEntry.flag == tt::Flag::kUpperBound && ttEntry.score <= alpha //
                || ttEntry.flag == tt::Flag::kLowerBound && ttEntry.score >= beta))
        {
            return ttEntry.score;
        }

        auto bestMove = kNullMove;
        auto bestScore = -kScoreInf;

        auto ttFlag = tt::Flag::kUpperBound;

        auto generator = MoveGenerator::main(pos, ttEntry.move);

        u32 legalMoves{};

        while (const auto move = generator.next()) {
            assert(pos.isPseudolegal(move));

            if constexpr (kRootNode) {
                if (!isLegalRootMove(move)) {
                    continue;
                }
                assert(pos.isLegal(move));
            } else if (!pos.isLegal(move)) {
                continue;
            }

            const auto baseLmr = s_lmrTable[depth][std::min<u32>(legalMoves, 63)];

            if constexpr (kPvNode) {
                curr.pv.length = 0;
            }

            ++legalMoves;

            const auto [newPos, guard] = thread.applyMove(pos, move);
            const auto sennichite = newPos.testSennichite(m_cuteChessWorkaround, thread.keyHistory);

            Score score;

            if (sennichite == SennichiteStatus::kWin) {
                // illegal perpetual
                continue;
            } else if (sennichite == SennichiteStatus::kDraw) {
                score = drawScore(thread.loadNodes());
            } else {
                const auto newDepth = depth - 1;

                if (depth >= 2 && legalMoves >= 5 + 2 * kRootNode && generator.stage() >= MovegenStage::NonCaptures) {
                    const auto reduced = std::min(std::max(newDepth - baseLmr, 1), newDepth - 1);
                    score = -search(thread, newPos, curr.pv, reduced, ply + 1, -alpha - 1, -alpha);

                    if (score > alpha && reduced < newDepth) {
                        score = -search(thread, newPos, curr.pv, newDepth, ply + 1, -alpha - 1, -alpha);
                    }
                } else if (!kPvNode || legalMoves > 1) {
                    score = -search(thread, newPos, curr.pv, depth - 1, ply + 1, -alpha - 1, -alpha);
                }

                if (kPvNode && (legalMoves == 1 || score > alpha)) {
                    score = -search<true>(thread, newPos, curr.pv, depth - 1, ply + 1, -beta, -alpha);
                }
            }

            if (hasStopped()) {
                return 0;
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
        }

        if (legalMoves == 0) {
            assert(!kRootNode);
            return -kScoreMate + ply;
        }

        m_ttable.put(pos.key(), bestScore, bestMove, depth, ply, ttFlag);

        return bestScore;
    }

    template <bool kPvNode>
    Score Searcher::qsearch(
        ThreadData& thread,
        const Position& pos,
        i32 ply,
        Score alpha,
        Score beta,
        Square captureSq
    ) {
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
            return pos.isInCheck() ? 0 : eval::staticEval(pos);
        }

        const auto staticEval = eval::staticEval(pos);

        if (staticEval >= beta) {
            return staticEval;
        }

        if (staticEval > alpha) {
            alpha = staticEval;
        }

        auto bestScore = staticEval;

        auto generator = MoveGenerator::qsearch(pos, captureSq);

        while (const auto move = generator.next()) {
            assert(pos.isPseudolegal(move));

            if (!pos.isLegal(move)) {
                continue;
            }

            const auto [newPos, guard] = thread.applyMove(pos, move);
            const auto sennichite = newPos.testSennichite(m_cuteChessWorkaround, thread.keyHistory);

            Score score;

            if (sennichite == SennichiteStatus::kWin) {
                // illegal perpetual
                continue;
            } else if (sennichite == SennichiteStatus::kDraw) {
                score = drawScore(thread.loadNodes());
            } else {
                score = -qsearch<kPvNode>(thread, newPos, ply + 1, -beta, -alpha, move.to());
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

    void Searcher::report(const ThreadData& bestThread, f64 time) {
        usize totalNodes = 0;
        i32 maxSeldepth = 0;

        for (const auto& thread : m_threads) {
            totalNodes += thread.loadNodes();
            maxSeldepth = std::max(maxSeldepth, thread.loadSeldepth());
        }

        protocol::DisplayScore score{};

        if (std::abs(bestThread.lastScore) >= kScoreMaxMate) {
            if (bestThread.lastScore > 0) {
                score = protocol::MateDisplayScore{kScoreMate - bestThread.lastScore};
            } else {
                score = protocol::MateDisplayScore{-(kScoreMate + bestThread.lastScore)};
            }
        } else {
            auto cp = bestThread.lastScore;

            // clamp draw scores to 0
            if (std::abs(cp) <= 2) {
                cp = 0;
            }

            score = protocol::CpDisplayScore{cp};
        }

        const protocol::SearchInfo info = {
            .depth = bestThread.depthCompleted,
            .seldepth = maxSeldepth,
            .timeSec = time,
            .nodes = totalNodes,
            .score = score,
            .pv = bestThread.lastPv,
            .hashfull = m_ttable.fullPermille(),
        };

        protocol::currHandler().printSearchInfo(std::cout, info);
    }

    void Searcher::finalReport(f64 time) {
        const auto& bestThread = m_threads[0];

        report(bestThread, time);
        protocol::currHandler().printBestMove(std::cout, bestThread.lastPv.moves[0]);
    }
} // namespace stoat
