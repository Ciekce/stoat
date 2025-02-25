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
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>

#include "../limit.h"
#include "../search.h"
#include "../util/ctrlc.h"
#include "format/stoatpack.h"

namespace stoat::datagen {
    namespace {
        constexpr usize kDatagenTtSizeMib = 16;

        std::atomic_bool s_stop{false};

        void initCtrlCHandler() {
            util::signal::addCtrlCHandler([] { s_stop.store(true); });
        }
    } // namespace

    i32 run(std::string_view output, u32 threads) {
        initCtrlCHandler();

        std::ofstream stream{std::string{output}, std::ios::binary};

        if (!stream) {
            std::cerr << "failed to open output file \"" << output << "\"" << std::endl;
            return 1;
        }

        Searcher searcher{kDatagenTtSizeMib};
        searcher.setLimiter(std::make_unique<limit::SoftNodeLimiter>(5000, 8388608));

        searcher.newGame();

        auto& thread = searcher.mainThread();
        thread.maxDepth = kMaxDepth;

        format::Stoatpack format{};

        auto pos = Position::startpos();

        std::vector<u64> keyHistory{};
        keyHistory.reserve(1024);

        format.startStandard();

        std::optional<format::Outcome> outcome{};

        std::cout << "Moves:";

        while (!outcome) {
            thread.reset(pos, keyHistory);
            searcher.runDatagenSearch();

            const auto blackScore = pos.stm() == Colors::kBlack ? thread.lastScore : -thread.lastScore;
            const auto move = thread.lastPv.moves[0];

            if (move.isNull()) {
                outcome = pos.stm() == Colors::kBlack ? format::Outcome::kBlackLoss : format::Outcome::kBlackWin;
                break;
            }

            if (std::abs(blackScore) > kScoreWin) {
                outcome = blackScore > 0 ? format::Outcome::kBlackWin : format::Outcome::kBlackLoss;
                break;
            }

            std::cout << ' ' << move;

            keyHistory.push_back(pos.key());
            pos = pos.applyMove(move);

            const auto sennichite = pos.testSennichite(false, keyHistory, 999999999);

            if (sennichite == SennichiteStatus::kDraw) {
                outcome = format::Outcome::kDraw;
                break;
            } else if (sennichite == SennichiteStatus::kWin) {
                std::cerr << "Illegal perpetual as best move?" << std::endl;
                return 1;
            }

            format.push(move, blackScore);
        }

        assert(outcome);

        std::cout << "\nOutcome: ";

        switch (*outcome) {
            case format::Outcome::kBlackLoss:
                std::cout << "Sente loss";
                break;
            case format::Outcome::kDraw:
                std::cout << "Draw";
                break;
            case format::Outcome::kBlackWin:
                std::cout << "Sente win";
                break;
        }

        std::cout << std::endl;

        format.writeAllWithOutcome(stream, *outcome);

        return 0;
    }
} // namespace stoat::datagen
