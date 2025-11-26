VERSION := $(file < version.txt)
DEFAULT_NET := $(file < network.txt)

ifndef EXE
    EXE = stoat-$(VERSION)
    override NO_EXE_SET = true
endif

ifndef EVALFILE
    EVALFILE = $(DEFAULT_NET).nnue
    override NO_EVALFILE_SET = true
endif

TYPE = native

# https://stackoverflow.com/a/1825832
rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

override SOURCES_3RDPARTY := 3rdparty/fmt/src/format.cc

override HEADERS := $(call rwildcard,src,*.h)
override SOURCES := $(call rwildcard,src,*.cpp)

# Sources including 3rdparty
override SOURCES_ALL := $(SOURCES) $(SOURCES_3RDPARTY)

CXX := clang++

CXXFLAGS := -I3rdparty/fmt/include -std=c++20 -flto -Wall -Wextra -Wno-sign-compare -fconstexpr-steps=4194304 -DST_NETWORK_FILE=\"$(EVALFILE)\" -DST_VERSION=$(VERSION)

CXXFLAGS_RELEASE := -O3 -DNDEBUG
CXXFLAGS_SANITIZER := -O1 -g -fsanitize=address,undefined

CXXFLAGS_NATIVE := -DST_NATIVE -march=native

ifdef NO_EXE_SET
    override EXE := $(EXE)-$(TYPE)
endif

LDFLAGS :=

COMPILER_VERSION := $(shell $(CXX) --version)

ifeq (, $(findstring clang,$(COMPILER_VERSION)))
    ifeq (, $(findstring gcc,$(COMPILER_VERSION)))
        ifeq (, $(findstring g++,$(COMPILER_VERSION)))
            $(error Only Clang and GCC supported)
        endif
    endif
endif

override MKDIR := mkdir -p

ifeq ($(OS), Windows_NT)
    override DETECTED_OS := Windows
    override SUFFIX := .exe
    override RM := del
    ifeq (.exe,$(findstring .exe,$(SHELL)))
        override MKDIR := -mkdir
    endif
else
    override DETECTED_OS := $(shell uname -s)
    override SUFFIX :=
    override RM := rm
    LDFLAGS += -pthread
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

override OUTFILE = $(subst .exe,,$(EXE))$(SUFFIX)

ifeq ($(TYPE), native)
    CXXFLAGS += $(CXXFLAGS_NATIVE) $(CXXFLAGS_RELEASE)
    BUILD_DIR = build-native
else ifeq ($(TYPE), sanitizer)
    CXXFLAGS += $(CXXFLAGS_NATIVE) $(CXXFLAGS_SANITIZER)
    BUILD_DIR = build-sanitizer
else
    $(error Unknown build type)
endif

override OBJECTS := $(addprefix $(BUILD_DIR)/,$(filter %.o,$(SOURCES_ALL:.cpp=.o) $(SOURCES_ALL:.cc=.o)))

define create_mkdir_target
$1:
	$(MKDIR) "$1"

.PRECIOUS: $1
endef

$(foreach dir,$(sort $(dir $(OBJECTS))),$(eval $(call create_mkdir_target,$(dir))))

ifeq ($(COMMIT_HASH),on)
    CXXFLAGS += -DST_COMMIT_HASH=$(shell git log -1 --pretty=format:%h)
endif

all: $(OUTFILE)

.PHONY: all

.DEFAULT_GOAL := $(OUTFILE)

ifdef NO_EVALFILE_SET
$(EVALFILE):
	$(info Downloading default network $(DEFAULT_NET).nnue)
	curl -sOL https://github.com/Ciekce/stoat-nets/releases/download/$(DEFAULT_NET)/$(DEFAULT_NET).nnue

download-net: $(EVALFILE)
endif

.SECONDEXPANSION:

$(BUILD_DIR)/%.o: %.cpp | $$(@D)/
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: %.cc | $$(@D)/
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OUTFILE): $(OBJECTS) $(EVALFILE)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(OUTFILE) $(OBJECTS)

bench: $(OUTFILE)
	./$(OUTFILE) bench

format: $(HEADERS) $(SOURCES)
	clang-format -i $^

