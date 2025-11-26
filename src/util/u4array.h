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

#pragma once

#include "../types.h"

#include <array>
#include <cassert>

namespace stoat::util {
    class IndexedU4 {
    public:
        constexpr inline operator u8() const { // NOLINT(google-explicit-constructor)
            return m_high ? (m_value >> 4) : (m_value & 0xF);
        }

        constexpr inline IndexedU4& operator=(u8 v) {
            assert(v <= 0xF);

            if (m_high) {
                m_value = (m_value & 0x0F) | (v << 4);
            } else {
                m_value = (m_value & 0xF0) | (v & 0x0F);
            }

            return *this;
        }

    private:
        constexpr IndexedU4(u8& value, bool high) :
                m_value{value}, m_high{high} {}

        u8& m_value;
        bool m_high;

        template <usize kSize>
        friend struct U4Array;
    };

    template <usize kSize>
    struct U4Array {
        static_assert(kSize % 2 == 0);

        std::array<u8, kSize / 2> data{};

        constexpr auto operator[](usize i) const {
            assert(i < kSize);
            return (data[i / 2] >> ((i % 2) * 4)) & 0xF;
        }

        constexpr auto operator[](usize i) {
            assert(i < kSize);
            return IndexedU4{data[i / 2], (i % 2) == 1};
        }
    };
} // namespace stoat::util
