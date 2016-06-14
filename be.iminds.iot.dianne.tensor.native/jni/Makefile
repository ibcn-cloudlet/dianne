ifndef TENSORNATIVE
TENSORNATIVE:=1
	# Determine this makefile's path.
	# Be sure to place this BEFORE `include` directives, if any.
	THIS_FILE := $(lastword $(MAKEFILE_LIST))
	
	# This file builds the native TH Tensor library and the JNI interface
	OS ?= $(shell uname -s)
	ARCH ?= $(shell uname -m)
	NATIVE ?= torch
	
	ifeq ($(OS),Linux)
	    include Makefile.Linux
	endif
	ifeq ($(OS),Darwin)
	    include Makefile.Darwin
	endif
	
	JAVA_INCLUDE := $(JAVA_HOME)/include
	JAVA_PLATFORMINCLUDE := $(JAVA_INCLUDE)/$(JDKPLATFORMINCLUDESUBDIR)
	INCLUDES := -I$(JAVA_INCLUDE) -I$(JAVA_PLATFORMINCLUDE)
	
	# Compiler config
	CCFLAGS += -O3 -fPIC
	DEPLOY_DIR ?= ../../native/$(NATIVE)/$(OS)/$(ARCH)/
	
	# export all variables: call export without arguments.
	export
endif

.PHONY: install
install:
	@$(MAKE) -f $(THIS_FILE) $(NATIVE);

.PHONY: cutorch
cutorch:
	$(MAKE) -C torch;
	$(MAKE) -C cutorch install;
	
.PHONY: torch
torch:
	$(MAKE) -C torch install;

.PHONY: clean cleanall
clean cleanall:
	$(MAKE) -C torch $(MAKECMDGOALS);
	$(MAKE) -C cutorch $(MAKECMDGOALS);
