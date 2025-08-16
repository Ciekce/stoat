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

#include "datagen.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <thread>

#include <fmt/ostream.h>
#include <fmt/std.h>

#include "../limit.h"
#include "../movegen.h"
#include "../search.h"
#include "../util/ctrlc.h"
#include "../util/rng.h"
#include "../util/static_vector.h"
#include "../util/timer.h"
#include "format/stoatpack.h"

namespace stoat::datagen {
    namespace {
        constexpr usize kDatagenTtSizeMib = 16;
        constexpr usize kReportInterval = 512;

        constexpr usize kBaseRandomMoves = 7;
        constexpr bool kRandomizeStartSide = true;

        constexpr usize kSoftNodes = 15000;
        constexpr usize kHardNodes = 8388608;

        std::mutex s_printMutex{};
        std::optional<std::ofstream> s_errOut{};

        std::atomic_bool s_stop{false};

        std::ofstream& getErrStream(const std::filesystem::path& outDir) {
            if (!s_errOut) {
                s_errOut = std::ofstream{outDir / "err.txt", std::ios::binary | std::ios::app};
            }

            return *s_errOut;
        }

        void initCtrlCHandler() {
            util::signal::addCtrlCHandler([] { s_stop.store(true); });
        }

        // NOTE: this does not test for entering kings
        [[nodiscard]] Move selectRandomLegal(
            util::rng::Jsf64Rng& rng,
            const Position& pos,
            std::vector<u64>& keyHistory,
            movegen::MoveList& moves
        ) {
            for (usize start = 0; start < moves.size(); ++start) {
                const auto idx = start + rng.nextU32(moves.size() - start);
                const auto move = moves[idx];

                if (!pos.isLegal(move)) {
                    continue;
                }

                keyHistory.push_back(pos.key());
                const auto newPos = pos.applyMove(move);
                const auto sennichite = newPos.testSennichite(false, keyHistory);
                keyHistory.pop_back();

                if (sennichite != SennichiteStatus::kWin) {
                    return move;
                }

                std::swap(moves[start], moves[idx]);
            }

            return kNullMove;
        }

        [[nodiscard]] Position getStartpos(
            util::rng::Jsf64Rng& rng,
            std::vector<u64>& keyHistory,
            format::IDataFormat& format
        ) {
            util::StaticVector<Move, kBaseRandomMoves + kRandomizeStartSide> randomMoves{};
            util::StaticVector<u64, kBaseRandomMoves + kRandomizeStartSide> newKeys{};

            Position pos{};

            const usize count = kBaseRandomMoves + (kRandomizeStartSide ? (rng.nextU64() >> 63) : 0);

            while (true) {
                randomMoves.clear();
                newKeys.clear();

                pos = Position::startpos();

                movegen::MoveList moves{};
                bool failed = false;

                for (usize i = 0; i < count; ++i) {
                    moves.clear();
                    movegen::generateAll(moves, pos);

                    const auto move = selectRandomLegal(rng, pos, keyHistory, moves);

                    if (!move) {
                        failed = true;
                        break;
                    }

                    randomMoves.push(move);
                    newKeys.push(pos.key());

                    pos = pos.applyMove(move);
                }

                if (!failed) {
                    break;
                }
            }

            std::ranges::copy(newKeys, std::back_inserter(keyHistory));

            for (const auto move : randomMoves) {
                format.pushUnscored(move);
            }

            return pos;
        }

        void runThread(u32 id, u64 seed, const std::filesystem::path& outDir) {
            const auto filename = fmt::format("{}.spk", id);
            const auto outFile = outDir / filename;

            std::ofstream stream{outFile, std::ios::binary | std::ios::app};

            if (!stream) {
                fmt::println(stderr, "failed to open output file \"{}\"", outFile);
                return;
            }

            util::rng::Jsf64Rng rng{seed};

            Searcher searcher{kDatagenTtSizeMib};
            searcher.setLimiter(std::make_unique<limit::SoftNodeLimiter>(kSoftNodes, kHardNodes));

            std::vector<u64> keyHistory{};
            keyHistory.reserve(1024);

            auto& thread = searcher.take();

            thread.maxDepth = kMaxDepth;
            thread.datagen = true;

            format::Stoatpack format{};

            usize gameCount{};
            usize totalPositions{};

            const auto start = util::Instant::now();

            const auto printProgress = [&] {
                const std::scoped_lock lock{s_printMutex};

                const auto time = start.elapsed();

                const auto gamesPerSec = static_cast<f64>(gameCount) / time;
                const auto posPerSec = static_cast<f64>(totalPositions) / time;

                fmt::println(
                    "thread {}: wrote {} positions from {} games in {:.6g} sec ({:.6g} games/sec, {:.6g} pos/sec)",
                    id,
                    totalPositions,
                    gameCount,
                    time,
                    gamesPerSec,
                    posPerSec
                );
            };

            while (!s_stop.load()) {
                searcher.newGame();

                format.startStandard();
                keyHistory.clear();

                auto pos = getStartpos(rng, keyHistory, format);
                thread.nnueState.reset(pos);

                u32 winPlies{};
                u32 lossPlies{};
                u32 drawPlies{};

                std::optional<format::Outcome> outcome{};

                while (!outcome) {
                    thread.reset(pos, keyHistory);
                    searcher.runDatagenSearch();

                    const auto& rootMove = thread.pvMove();

                    const auto blackScore = pos.stm() == Colors::kBlack ? rootMove.score : -rootMove.score;
                    const auto move = rootMove.pv.moves[0];

                    if (move.isNull()) {
                        outcome =
                            pos.stm() == Colors::kBlack ? format::Outcome::kBlackLoss : format::Outcome::kBlackWin;
                        break;
                    }

                    if (std::abs(blackScore) > kScoreWin) {
                        outcome = blackScore > 0 ? format::Outcome::kBlackWin : format::Outcome::kBlackLoss;
                        break;
                    }

                    const auto oldPos = pos;

                    keyHistory.push_back(pos.key());
                    pos = pos.applyMove<NnueUpdateAction::kApplyInPlace>(move, &thread.nnueState);

                    const auto sennichite = pos.testSennichite(false, keyHistory, 999999999);

                    if (sennichite == SennichiteStatus::kDraw) {
                        outcome = format::Outcome::kDraw;
                        break;
                    } else if (sennichite == SennichiteStatus::kWin) {
                        const std::scoped_lock lock{s_printMutex};

                        auto& errStream = getErrStream(outDir);

                        fmt::println(errStream, "thread {}: illegal perpetual as best move?", id);

                        fmt::print(errStream, "Keys:");
                        for (usize i = 0; i < keyHistory.size() - 1; ++i) {
                            fmt::print(errStream, " {:016x}", keyHistory[i]);
                        }
                        fmt::println(errStream, "");

                        fmt::println(errStream, "Pos: {}", oldPos.sfen());
                        fmt::println(errStream, "Move: {}", move);

                        errStream.flush();

                        outcome = format::Outcome::kDraw;
                        break;
                    }

                    // This will likely be handled by search returning
                    // a mate score, but test for it just in case
                    if (pos.isEnteringKingsWin()) {
                        outcome =
                            pos.stm() == Colors::kBlack ? format::Outcome::kBlackWin : format::Outcome::kBlackLoss;
                        break;
                    }

                    if (blackScore >= 1000) {
                        ++winPlies;
                        lossPlies = 0;
                        drawPlies = 0;
                    } else if (blackScore <= -1000) {
                        winPlies = 0;
                        ++lossPlies;
                        drawPlies = 0;
                    } else if (pos.moveCount() >= 40 && std::abs(blackScore) <= 10) {
                        winPlies = 0;
                        lossPlies = 0;
                        ++drawPlies;
                    } else {
                        winPlies = 0;
                        lossPlies = 0;
                        drawPlies = 0;
                    }

                    if (winPlies >= 6) {
                        outcome = format::Outcome::kBlackWin;
                    } else if (lossPlies >= 6) {
                        outcome = format::Outcome::kBlackLoss;
                    } else if (drawPlies >= 10) {
                        outcome = format::Outcome::kDraw;
                    }

                    format.push(move, blackScore);
                }

                assert(outcome);

                totalPositions += format.writeAllWithOutcome(stream, *outcome);
                stream.flush();

                ++gameCount;

                if ((gameCount % kReportInterval) == 0) {
                    printProgress();
                }
            }

            if ((gameCount % kReportInterval) != 0) {
                printProgress();
            }
        }
    } // namespace

    i32 run(std::string_view output, u32 threadCount) {
        initCtrlCHandler();

        const auto outDir = std::filesystem::path{output};

        if (!std::filesystem::exists(outDir)) {
            std::filesystem::create_directories(outDir);
        }

        if (!std::filesystem::is_directory(outDir)) {
            fmt::println(stderr, "out path must be a directory");
            return 1;
        }

        const auto baseSeed = util::rng::generateSingleSeed();
        fmt::println("Base seed: {}", baseSeed);

        util::rng::SeedGenerator seedGenerator{baseSeed};

        fmt::println("Starting {} threads", threadCount);

        std::vector<std::thread> threads{};
        threads.reserve(threadCount);

        for (u32 id = 0; id < threadCount; ++id) {
            const auto seed = seedGenerator.nextSeed();
            threads.emplace_back([&, id, seed] { runThread(id, seed, outDir); });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        if (s_errOut) {
            s_errOut = {};
        }

        fmt::println("done");

        return 0;
    }
} // namespace stoat::datagen
