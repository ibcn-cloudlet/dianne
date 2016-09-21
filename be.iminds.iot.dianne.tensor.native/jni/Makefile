# Determine this makefile's path.
# Be sure to place this BEFORE `include` directives, if any.
THIS_FILE := $(lastword $(MAKEFILE_LIST))

# This file builds the native TH Tensor library and the JNI interface
export NATIVE ?= torch

.PHONY: install
install:
	@$(MAKE) -f $(THIS_FILE) $(NATIVE);

.PHONY: cudnn
cudnn:
	$(MAKE) -C torch;
#	$(MAKE) -C cutorch;
	$(MAKE) -C cudnn install;

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
