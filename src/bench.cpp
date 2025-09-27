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
#include "stats.h"

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
            "l1r6/4S2k1/p2pb1gsg/5pp2/1pBnp3p/5PP2/PP1P1G1S1/6GKL/L1R5L b Ps3n5p 93"sv,
            "5+P+B+R1/1kg2+P1+P+R/1g1s2KG1/3g4p/2p1pS3/1+p+l1s4/4B1N1P/9/4P4 b S3N3L9P 221"sv,
            "ln3g1nl/1r1sg1sk1/p1p1ppbp1/1p1p2p1p/2P6/3P4P/PP2PPPP1/1BRS2SK1/LNG2G1NL b - 23"sv,
            "l1+R4nk/5rgs1/3pp1gp1/p4pp1l/1p5Pp/4PSP2/P4PNG1/4G4/L5K1L w 2BP2s2n4p 88"sv,
            "6B1+S/2gg5/4lp1+P1/6p1p/4pP1R1/Ppk1P1P1P/2+p2GK2/5S3/1+n3+r2L b B2SN2L2Pg2n4p 149"sv,
            "7nl/3+P1kg2/4pb1ps/2r2NP1p/l1P2P1P1/s7P/PN2P4/KGB2G3/1N1R4L w G5P2sl2p 98"sv,
            "l4Grnl/1B2+B1gk1/p1n3sp1/4ppp1p/P1S2P1P1/1PGP2P1P/3pP2g1/1K4sR1/LN6L w 3Psn 78"sv,
            "ln6l/2gkgr1s1/1p1pp1n1p/3s1pP2/p8/1P1PBPb2/PS2P1NpP/1K1G2R2/LN1G4L w 3Psp 58"sv,
            "ln1gk2nl/1rs3g2/p3ppspp/2pp2p2/1p5PP/2P6/PPSPPPP2/2G3SR1/LN2KG1NL b Bb 21"sv,
            "ln7/1r2g1g2/2pspk1bn/pp1p2PB1/5pp1p/P1P2P3/1PSPP3+l/3K2S2/LN1G1G3 b Srnl3p 59"sv,
            "4g2nl/5skn1/p1pppp1p1/6p+b1/4P4/3+R1SL1p/P3GPPP1/1+r2SS1KP/3PL2NL w GPbgn2p 128"sv,
            "lnsgk2nl/1r4gs1/p1pppp1pp/6p2/1p5P1/2P6/PPSPPPP1P/7R1/LN1GKGSNL b Bb 13"sv,
            "ln1g1gsnl/1r1s2k2/p1pp1p1p1/6p1p/1p7/2P5P/PPS+b1PPP1/2B3K2/LN1GRGSNL w P2p 26"sv,
            "l2sk2nl/2g2s1g1/2n1pp1pp/pr4p2/1p6P/P2+b+RP1P1/1P2PSP2/5K3/L2G1G1NL b SPbn3p 51"sv,
        };

        constexpr usize kTtSizeMib = 16;
    } // namespace

    void run(i32 depth) {
        Searcher searcher{kTtSizeMib};
        searcher.ensureReady();

        usize totalNodes{};
        f64 totalTime{};

        auto& thread = searcher.take();

        thread.maxDepth = depth;

        for (const auto sfen : kBenchSfens) {
            fmt::println("SFEN: {}", sfen);

            thread.reset(Position::fromSfen(sfen).take(), {});

            BenchInfo info{};
            searcher.runBenchSearch(info);

            totalNodes += info.nodes;
            totalTime += info.time;

            fmt::println("");
        }

        const auto nps = static_cast<usize>(static_cast<f64>(totalNodes) / totalTime);

        fmt::println("{:.5g} seconds", totalTime);
        fmt::println("{} nodes {} nps", totalNodes, nps);

        stats::print();
    }
} // namespace stoat::bench
