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

#include "eval/eval.h"
#include "protocol/handler.h"

namespace stoat {
    namespace {
        void generateLegal(movegen::MoveList& dst, const Position& pos) {
            movegen::MoveList generated{};
            movegen::generateAll(generated, pos);

            for (const auto move : generated) {
                if (pos.isLegal(move)) {
                    dst.push(move);
                }
            }
        }

        inline Score drawScore(usize nodes) {
            return 2 - static_cast<Score>(nodes % 4);
        }
    } // namespace

    ThreadData::ThreadData() {
        keyHistory.reserve(1024);
        stack.resize(kMaxDepth);
    }

    void ThreadData::reset(const Position& newRootPos, std::span<const u64> newKeyHistory) {
        rootPos = newRootPos;

        keyHistory.clear();
        keyHistory.reserve(newKeyHistory.size());

        std::ranges::copy(newKeyHistory, std::back_inserter(keyHistory));

        stats.seldepth.store(0);
        stats.nodes.store(0);
    }

    std::pair<Position, ThreadPosGuard> ThreadData::applyMove(const Position& pos, Move move) {
        keyHistory.push_back(pos.key());
        return std::pair<Position, ThreadPosGuard>{
            std::piecewise_construct,
            std::forward_as_tuple(pos.applyMove(move)),
            std::forward_as_tuple(keyHistory)
        };
    }

    Searcher::Searcher() {
        setThreads(1);
    }

    Searcher::~Searcher() {
        stop();
        stopThreads();
    }

    void Searcher::newGame() {
        //
    }

    void Searcher::ensureReady() {
        //
    }

    void Searcher::setThreads(u32 threadCount) {
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

    void Searcher::startSearch(
        const Position& pos,
        std::span<const u64> keyHistory,
        util::Instant startTime,
        bool infinite,
        i32 maxDepth,
        std::unique_ptr<limit::ISearchLimiter> limiter
    ) {
        m_resetBarrier.arriveAndWait();

        const std::unique_lock lock{m_searchMutex};

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

            const auto score = search<true>(thread, thread.rootPos, rootPv, depth, 0, -kScoreInf, kScoreInf);

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

    template <bool kRootNode>
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

        if (!kRootNode && thread.isMainThread() && thread.rootDepth > 1) {
            if (m_limiter->stopHard(thread.loadNodes())) {
                m_stop.store(true, std::memory_order::relaxed);
                return 0;
            }
        }

        thread.incNodes();

        if (depth <= 0) {
            return eval::staticEval(pos);
        }

        thread.updateSeldepth(ply + 1);

        if (ply >= kMaxDepth) {
            return pos.isInCheck() ? 0 : eval::staticEval(pos);
        }

        auto& curr = thread.stack[ply];

        auto bestScore = -kScoreInf;

        movegen::MoveList moves{};
        movegen::generateAll(moves, pos);

        u32 legalMoves{};

        for (const auto move : moves) {
            if constexpr (kRootNode) {
                if (!isLegalRootMove(move)) {
                    continue;
                }
                assert(pos.isLegal(move));
            } else if (!pos.isLegal(move)) {
                continue;
            }

            curr.pv.length = 0;

            ++legalMoves;

            const auto [newPos, guard] = thread.applyMove(pos, move);
            const auto sennichite = newPos.testSennichite(thread.keyHistory);

            Score score{};

            if (sennichite == SennichiteStatus::kWin) {
                // illegal perpetual
                continue;
            } else if (sennichite == SennichiteStatus::kDraw) {
                score = drawScore(thread.loadNodes());
            } else {
                score = -search(thread, newPos, curr.pv, depth - 1, ply + 1, -beta, -alpha);
            }

            if (hasStopped()) {
                return 0;
            }

            if (score > bestScore) {
                bestScore = score;
            }

            if (score > alpha) {
                alpha = score;

                assert(curr.pv.length + 1 <= kMaxDepth);
                pv.update(move, curr.pv);

                if (score >= beta) {
                    break;
                }
            }
        }

        if (legalMoves == 0) {
            assert(!kRootNode);
            return -kScoreMate + ply;
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
        };

        protocol::currHandler().printSearchInfo(std::cout, info);
    }

    void Searcher::finalReport(f64 time) {
        const auto& bestThread = m_threads[0];

        report(bestThread, time);
        protocol::currHandler().printBestMove(std::cout, bestThread.lastPv.moves[0]);
    }
} // namespace stoat
