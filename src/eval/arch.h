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

namespace stoat::eval {
    constexpr i32 kFtQBits = 8;
    constexpr i32 kL1QBits = 7;
    constexpr i32 kQBits = 6;

    constexpr i32 kFtScaleBits = 7;

    constexpr u32 kFtSize = 2344;

    constexpr u32 kL1Size = 1536;
    constexpr u32 kL2Size = 16;
    constexpr u32 kL3Size = 32;

    constexpr i32 kScale = 400;
} // namespace stoat::eval
