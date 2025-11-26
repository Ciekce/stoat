cmake_minimum_required(VERSION 3.15)

file(STRINGS version.txt ST_VERSION LIMIT_COUNT 1)
project(stoat VERSION ${ST_VERSION})

set(CMAKE_CXX_STANDARD 20)

file(STRINGS network.txt ST_DEFAULT_NET_NAME LIMIT_COUNT 1)

set(ST_CLANG CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

if(MSVC AND NOT ST_CLANG)
	message(FATAL_ERROR Stoat does not support building with MSVC)
endif()

if(MSVC)
	add_compile_options(/EHsc /W4 /clang:-Wno-sign-compare /WX)
else()
	add_compile_options(-Wall -Wextra -Wno-sign-compare -Werror)
endif()

option(ST_FAST_PEXT "whether pext and pdep are usably fast on this architecture" ON)

add_executable(stoat-native 3rdparty/fmt/src/format.cc src/main.cpp src/types.h src/core.h src/bitboard.h
	src/util/bits.h src/position.h src/position.cpp src/util/result.h src/util/split.h src/util/split.cpp
	src/util/parse.h src/move.h src/util/string_map.h src/attacks/attacks.h src/util/multi_array.h src/movegen.h
	src/util/static_vector.h src/movegen.cpp src/perft.h src/perft.cpp src/util/timer.h src/util/timer.cpp src/arch.h
	src/attacks/sliders/util.h src/attacks/sliders/data.h src/attacks/sliders/bmi2.h src/attacks/sliders/bmi2.cpp
	src/rays.h src/keys.h src/util/rng.h src/protocol/handler.h src/protocol/handler.cpp src/protocol/uci_like.h
	src/protocol/uci_like.cpp src/pv.h src/protocol/common.h src/protocol/usi.h src/protocol/usi.cpp src/protocol/uci.h
	src/protocol/uci.cpp src/search.h src/search.cpp src/util/barrier.h src/eval/eval.h src/eval/eval.cpp src/limit.h
	src/limit.cpp src/bench.h src/bench.cpp src/thread.h src/thread.cpp src/attacks/sliders/magics.h
	src/attacks/sliders/black_magic.h src/attacks/sliders/black_magic.cpp src/ttable.h src/ttable.cpp src/util/align.h
	src/util/range.h src/movepick.h src/movepick.cpp src/see.h src/see.cpp src/datagen/format/format.h
	src/datagen/format/stoatpack.h src/datagen/format/stoatpack.cpp src/datagen/format/stoatformat.h
	src/datagen/format/stoatformat.cpp src/util/u4array.h src/datagen/datagen.h src/datagen/datagen.cpp src/util/ctrlc.h
	src/util/ctrlc.cpp src/eval/arch.h src/eval/nnue.h src/eval/nnue.cpp src/history.h src/history.cpp src/stats.h
	src/stats.cpp src/correction.h src/correction.cpp
)

target_include_directories(stoat-native PUBLIC 3rdparty/fmt/include)
target_compile_options(stoat-native PUBLIC -march=native $<$<CONFIG:Release>:-flto>)
target_link_options(stoat-native PUBLIC -fuse-ld=lld)
target_compile_definitions(stoat-native PUBLIC ST_NATIVE ST_VERSION=${CMAKE_PROJECT_VERSION}
	ST_NETWORK_FILE="${PROJECT_SOURCE_DIR}/${ST_DEFAULT_NET_NAME}.nnue")

if(MSVC)
	target_compile_options(stoat-native PUBLIC /clang:-fconstexpr-steps=4194304)
else()
	target_compile_options(stoat-native PUBLIC -fconstexpr-steps=4194304)
endif()

if(ST_FAST_PEXT)
	target_compile_definitions(stoat-native PUBLIC ST_FAST_PEXT)
endif()
