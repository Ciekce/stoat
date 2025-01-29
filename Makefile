VERSION := $(file < version.txt)

ifndef EXE
    EXE = stoat-$(VERSION)
    NO_EXE_SET = true
endif

SOURCES := src/main.cpp src/position.cpp src/util/split.cpp src/move.cpp src/movegen.cpp src/perft.cpp src/util/timer.cpp src/attacks/sliders/bmi2.cpp src/protocol/handler.cpp src/protocol/uci_like.cpp src/protocol/usi.cpp src/protocol/uci.cpp src/search.cpp src/eval/eval.cpp src/limit.cpp src/bench.cpp src/thread.cpp src/attacks/sliders/black_magic.cpp src/ttable.cpp src/movepick.cpp src/history.cpp

SUFFIX :=

CXX := clang++

CXXFLAGS := -std=c++20 -flto -fconstexpr-steps=4194304 -DST_VERSION=$(VERSION)

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
    $(CXX) $(CXXFLAGS) $(CXXFLAGS_$1) $(CXXFLAGS_$2) $(LDFLAGS) -o $(EXE)$(if $(NO_EXE_SET),-$3)$(SUFFIX) $^
endef

all: native

.PHONY: all

.DEFAULT_GOAL := native

$(EXE): $(SOURCES)
	$(call build,NATIVE,RELEASE,native)

native: $(EXE)

sanitizer: $(SOURCES)
	$(call build,NATIVE,SANITIZER,native)

clean:

