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

#include "types.h"

#include <new>

#if defined(ST_NATIVE)
    // cannot expand a macro to defined()
    #if __BMI2__ && defined(ST_FAST_PEXT)
        #define ST_HAS_FAST_PEXT 1
    #else
        #define ST_HAS_FAST_PEXT 0
    #endif
#else //TODO others
    #error no arch specified
#endif

namespace stoat {
#ifdef __cpp_lib_hardware_interference_size
    constexpr auto kCacheLineSize = std::hardware_destructive_interference_size;
#else
    constexpr usize kCacheLineSize = 64;
#endif
} // namespace stoat
