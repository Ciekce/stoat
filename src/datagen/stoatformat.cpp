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

#include "stoatformat.h"

namespace stoat::datagen::format {
    Color StoatformatRecord::stm() const {
        const auto stm = static_cast<u8>(occ[0] >> 90) & 0x1;
        return Color::fromRaw(stm);
    }

    void StoatformatRecord::setStm(Color stm) {
        static constexpr u128 kStmMask = u128{0x1} << 90;
        occ[0] = (occ[0] & ~kStmMask) | (static_cast<u128>(stm.raw()) << 90);
    }

    Outcome StoatformatRecord::wdl() const {
        const auto wdl = static_cast<u32>(occ[0] >> 88) & 0x3;
        assert(wdl <= 2);
        return static_cast<Outcome>(wdl);
    }

    void StoatformatRecord::setWdl(Outcome wdl) {
        static constexpr u128 kWdlMask = u128{0x3} << 88;
        occ[0] = (occ[0] & ~kWdlMask) | (static_cast<u128>(wdl) << 88);
    }

    StoatformatRecord StoatformatRecord::pack(const Position& pos, i16 senteScore, Outcome wdl) {
        StoatformatRecord record{};

        record.occ[0] = pos.colorBb(Colors::kBlack).raw();
        record.occ[1] = pos.colorBb(Colors::kWhite).raw();

        const auto blackHand = pos.hand(Colors::kBlack);
        const auto whiteHand = pos.hand(Colors::kWhite);

        record.occ[0] |= static_cast<u128>(blackHand.raw()) << 96;
        record.occ[1] |= static_cast<u128>(whiteHand.raw()) << 96;

        const auto stm = static_cast<u128>(pos.stm().raw());
        const auto stmWdl = (stm << 90) | (static_cast<u128>(wdl) << 88);
        record.occ[0] |= stmWdl;

        usize pieceIdx = 0;

        auto blackPieces = pos.colorBb(Colors::kBlack);
        while (!blackPieces.empty()) {
            const auto sq = blackPieces.popLsb();
            const auto pt = pos.pieceOn(sq).type();
            record.pieces[pieceIdx++] = pt.raw();
        }

        auto whitePieces = pos.colorBb(Colors::kWhite);
        while (!whitePieces.empty()) {
            const auto sq = whitePieces.popLsb();
            const auto pt = pos.pieceOn(sq).type();
            record.pieces[pieceIdx++] = pt.raw();
        }

        record.score = senteScore;
        record.plyCount = pos.moveCount();

        return record;
    }
} // namespace stoat::datagen::format
