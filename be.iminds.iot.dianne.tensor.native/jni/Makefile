# Determine this makefile's path.
# Be sure to place this BEFORE `include` directives, if any.
THIS_FILE := $(lastword $(MAKEFILE_LIST))

# This file builds the native TH Tensor library and the JNI interface
OS ?= $(shell uname -s)
ARCH ?= $(shell uname -m)
# basic check to see if any CUDA compatible GPU is installed
# set this to false to turn off GPU related functionality

HAS_GPU ?= $(shell lspci -v 2>/dev/null | grep NVIDIA && echo true)
HAS_NVCC ?= $(shell nvcc --version > /dev/null 2>&1 && echo true)
NATIVE ?= torch

ifeq ($(OS),Linux)
    include Makefile.Linux
endif
ifeq ($(OS),Darwin)
    include Makefile.Darwin
    OS := MacOSX
endif

JAVA_INCLUDE := $(JAVA_HOME)/include
JAVA_PLATFORMINCLUDE := $(JAVA_INCLUDE)/$(JDKPLATFORMINCLUDESUBDIR)
INCLUDES := -I$(JAVA_INCLUDE) -I$(JAVA_PLATFORMINCLUDE)

# Compiler config
CCFLAGS := -O3 -fPIC
DEPLOY_DIR ?= $(abspath ../native/$(NATIVE)/$(OS)/$(ARCH)/)

# export all variables: call export without arguments.
export

.PHONY: install
install:
	@$(MAKE) -f $(THIS_FILE) $(NATIVE);

.PHONY: cutorch
cutorch: torch
	NATIVE=cutorch $(MAKE) -C cutorch install;
	
.PHONY: torch
torch:
	NATIVE=torch $(MAKE) -C torch install;

.PHONY: clean cleanall
clean cleanall:
	$(MAKE) -C $(NATIVE) $(MAKECMDGOALS);