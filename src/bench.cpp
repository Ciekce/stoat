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

#include "bench.h"

#include <array>
#include <string_view>

#include "position.h"
#include "search.h"

namespace stoat::bench {
    namespace {
        using namespace std::string_view_literals;

        // partially from the USI spec, partially from YaneuraOu
        constexpr std::array kBenchSfens = {
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"sv,
            "8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p 124"sv,
            "lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9"sv,
            "l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1"sv,
            "6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1"sv,
            "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1"sv,
        };
    } // namespace

    void run(i32 depth) {
        Searcher searcher{};
        searcher.newGame();

        usize totalNodes{};
        f64 totalTime{};

        for (const auto sfen : kBenchSfens) {
            std::cout << "SFEN: " << sfen << std::endl;

            const auto pos = Position::fromSfen(sfen).take();

            BenchInfo info{};
            searcher.runBenchSearch(info, pos, depth);

            totalNodes += info.nodes;
            totalTime += info.time;

            std::cout << std::endl;
        }

        const auto nps = static_cast<usize>(static_cast<f64>(totalNodes) / totalTime);

        std::cout << totalTime << " seconds" << std::endl;
        std::cout << totalNodes << " nodes " << nps << " nps" << std::endl;
    }
} // namespace stoat::bench
