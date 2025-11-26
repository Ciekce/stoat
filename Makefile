VERSION := $(file < version.txt)
DEFAULT_NET := $(file < network.txt)

ifndef EXE
    EXE = stoat-$(VERSION)
    NO_EXE_SET = true
endif

ifndef EVALFILE
    EVALFILE = $(DEFAULT_NET).nnue
    NO_EVALFILE_SET = true
endif

SOURCES := src/3rdparty/fmt/src/format.cc src/main.cpp src/position.cpp src/util/split.cpp src/movegen.cpp src/perft.cpp src/util/timer.cpp src/attacks/sliders/bmi2.cpp src/protocol/handler.cpp src/protocol/uci_like.cpp src/protocol/usi.cpp src/protocol/uci.cpp src/search.cpp src/eval/eval.cpp src/limit.cpp src/bench.cpp src/thread.cpp src/attacks/sliders/black_magic.cpp src/ttable.cpp src/movepick.cpp src/see.cpp src/datagen/format/stoatpack.cpp src/datagen/format/stoatformat.cpp src/datagen/datagen.cpp src/util/ctrlc.cpp src/eval/nnue.cpp src/history.cpp src/stats.cpp src/correction.cpp

SUFFIX :=

CXX := clang++

CXXFLAGS := -Isrc/3rdparty/fmt/include -std=c++20 -flto -Wall -Wextra -Wno-sign-compare -fconstexpr-steps=4194304 -DST_NETWORK_FILE=\"$(EVALFILE)\" -DST_VERSION=$(VERSION)

CXXFLAGS_RELEASE := -O3 -DNDEBUG
CXXFLAGS_SANITIZER := -O1 -g -fsanitize=address -fsanitize=undefined

CXXFLAGS_NATIVE := -DST_NATIVE -march=native

LDFLAGS :=

COMPILER_VERSION := $(shell $(CXX) --version)

ifeq (, $(findstring clang,$(COMPILER_VERSION)))
    ifeq (, $(findstring gcc,$(COMPILER_VERSION)))
        ifeq (, $(findstring g++,$(COMPILER_VERSION)))
            $(error Only Clang and GCC supported)
        endif
    endif
endif

ifeq ($(OS), Windows_NT)
    DETECTED_OS := Windows
    SUFFIX := .exe
    RM := del
else
    DETECTED_OS := $(shell uname -s)
    LDFLAGS += -pthread
    RM := rm
endif

ifneq (, $(findstring clang,$(COMPILER_VERSION)))
    ifneq ($(DETECTED_OS),Darwin)
        LDFLAGS += -fuse-ld=lld
    endif
endif

ARCH_DEFINES := $(shell echo | $(CXX) -march=native -E -dM -)

ifneq ($(findstring __BMI2__, $(ARCH_DEFINES)),)
    ifeq ($(findstring __znver1, $(ARCH_DEFINES)),)
        ifeq ($(findstring __znver2, $(ARCH_DEFINES)),)
            ifeq ($(findstring __bdver, $(ARCH_DEFINES)),)
                CXXFLAGS_NATIVE += -DST_FAST_PEXT
            endif
        endif
    endif
endif

ifeq ($(COMMIT_HASH),on)
    CXXFLAGS += -DST_COMMIT_HASH=$(shell git log -1 --pretty=format:%h)
endif

define build
    $(CXX) $(CXXFLAGS) $(CXXFLAGS_$1) $(CXXFLAGS_$2) $(LDFLAGS) -o $(EXE)$(if $(NO_EXE_SET),-$3)$(SUFFIX) $(filter-out $(EVALFILE),$^)
endef

all: native

.PHONY: all

.DEFAULT_GOAL := native

ifdef NO_EVALFILE_SET
$(EVALFILE):
	$(info Downloading default network $(DEFAULT_NET).nnue)
	curl -sOL https://github.com/Ciekce/stoat-nets/releases/download/$(DEFAULT_NET)/$(DEFAULT_NET).nnue

download-net: $(EVALFILE)
endif

$(EXE): $(EVALFILE) $(SOURCES)
	$(call build,NATIVE,RELEASE,native)

native: $(EXE)

bench: native
	./$(EXE)$(if $(NO_EXE_SET),-native)$(SUFFIX) bench

sanitizer: $(EVALFILE) $(SOURCES)
	$(call build,NATIVE,SANITIZER,native)

clean:

