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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>

namespace stoat::util {
    template <typename T, usize Capacity>
    class StaticVector {
    public:
        StaticVector() = default;

        StaticVector(const StaticVector<T, Capacity>& other) {
            *this = other;
        }

        inline void push(const T& elem) {
            assert(m_size < Capacity);
            m_data[m_size++] = elem;
        }

        inline void push(T&& elem) {
            assert(m_size < Capacity);
            m_data[m_size++] = std::move(elem);
        }

        inline bool tryPush(const T& elem) {
            if (m_size < Capacity) {
                push(elem);
                return true;
            }

            return false;
        }

        inline bool tryPush(T&& elem) {
            if (m_size < Capacity) {
                push(elem);
                return true;
            }

            return false;
        }

        inline void clear() {
            m_size = 0;
        }

        inline void fill(const T& v) {
            m_data.fill(v);
        }

        [[nodiscard]] inline usize size() const {
            return m_size;
        }

        [[nodiscard]] inline bool empty() const {
            return m_size == 0;
        }

        [[nodiscard]] inline const T& operator[](usize i) const {
            assert(i < m_size);
            return m_data[i];
        }

        [[nodiscard]] inline auto begin() {
            return m_data.begin();
        }

        [[nodiscard]] inline auto end() {
            return m_data.begin() + static_cast<std::ptrdiff_t>(m_size);
        }

        [[nodiscard]] inline T& operator[](usize i) {
            assert(i < m_size);
            return m_data[i];
        }

        [[nodiscard]] inline auto begin() const {
            return m_data.begin();
        }

        [[nodiscard]] inline auto end() const {
            return m_data.begin() + static_cast<std::ptrdiff_t>(m_size);
        }

        [[nodiscard]] inline auto cbegin() const {
            return m_data.cbegin();
        }

        [[nodiscard]] inline auto cend() const {
            return m_data.cbegin() + static_cast<std::ptrdiff_t>(m_size);
        }

        inline void resize(usize size) {
            assert(size <= Capacity);
            m_size = size;
        }

        inline StaticVector<T, Capacity>& operator=(const StaticVector<T, Capacity>& other) {
            std::copy(other.begin(), other.end(), begin());
            m_size = other.m_size;
            return *this;
        }

    private:
        std::array<T, Capacity> m_data{};
        usize m_size{0};
    };
} // namespace stoat::util
