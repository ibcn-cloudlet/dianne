include ../Makefile.system

CUDA_HOME ?= /usr/local/cuda
ifeq ($(strip $(ARCH)),x86_64)
    CUDA_TARGET:=${CUDA_HOME}/targets/x86_64-linux
else ifeq ($(findstring arm,$(ARCH)),arm)
    CUDA_TARGET:=${CUDA_HOME}/targets/armv7-linux-gnueabihf
else ifeq ($(strip $(ARCH)),aarch64)
    CUDA_TARGET:=${CUDA_HOME}/targets/aarch64-linux
else
    CUDA_TARGET:=${CUDA_HOME}
endif

ifneq (,$(filter $(ARCH),x86_64 aarch64))
    TARGET_SIZE := 64
else ifeq ($(findstring arm,$(ARCH)),arm)
    TARGET_SIZE := 32
else
    $(info ERROR - unsupported value $(ARCH) for ARCH!)
endif

ifeq ($(wildcard $(CUDA_TARGET)),)
    CUDA_TARGET:=${CUDA_HOME}
endif

CUDA_LIB:=lib
ifeq ($(wildcard $(CUDA_TARGET)/$(CUDA_LIB)),)
    CUDA_LIB:=lib64
endif

HEADERS := $(wildcard ../torch/*.h)
SRC := $(wildcard ../torch/*.c)
OBJECTS := $(notdir $(SRC:.c=.o))
JAVAH := ../../../be.iminds.iot.dianne.tensor/generated/jni-headers
LINKS := -Lcunn/lib/THCUNN/build -lTHCUNN -Lcutorch/lib/THC/build -lTHC -L../torch/torch7/lib/TH/build -lTH -L$(CUDA_TARGET)/$(CUDA_LIB) -lcudart_static -lcublas_static -lstdc++ -lculibos -pthread
INCLUDES += -I. -I../torch
INCLUDES += -I$(CUDA_TARGET)/include/
INCLUDES += -Icutorch/lib -Icutorch/lib/THC -Icutorch/lib/THC/build
INCLUDES += -Icunn/lib/THCUNN -Icunn/lib/THCUNN/build
INCLUDES += -I../torch/torch7/lib/TH -I../torch/torch7/lib/TH/build
INCLUDES += -I../torch/nn/lib/THNN -I../torch/nn/lib/THNN/build
INCLUDES += -I$(JAVAH)
CCFLAGS += -DCUDA

NVCC=nvcc -ccbin $(CXX)
export NVCC
NVCC_FLAGS := -DCUDA -Xcompiler -fPIC -m${TARGET_SIZE}
CUDA_HEADERS := $(wildcard *.h)
CUDA_SRC := $(wildcard *.cu)
CUDA_OBJECTS := $(CUDA_SRC:.cu=.o)

# CUDA code generation flags
CUDA_ARCHS := 2.0 3.0 3.2 3.5 3.7 5.0 5.2 5.3
ifneq ($(strip $(CUDA_ARCH_BIN)),)
    CUDA_ARCHS := $(CUDA_ARCH_BIN)
endif
CUDA_ARCHS := $(subst .,,$(CUDA_ARCHS))
GEN_CODES += $(foreach CUDA_ARCH,$(CUDA_ARCHS),-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH))
GEN_CODES += -gencode arch=compute_$(lastword $(CUDA_ARCHS)),code=compute_$(lastword $(CUDA_ARCHS))

ifneq ($(strip $(OS))-$(strip $(ARCH)),$(shell uname -s)-$(shell uname -m))
#sets CROSS_COMPILATION to TRUE
	CMAKE_FLAGS += -DCMAKE_SYSTEM_NAME=$(OS) -DCMAKE_SYSTEM_PROCESSOR=$(ARCH)
endif

.PHONY: info install
install: info $(DEPLOY_DIR)/$(LIB)

info:
	$(info CUDA_TARGET: $(CUDA_TARGET) CUDA_LIB: $(CUDA_LIB))

$(DEPLOY_DIR)/$(LIB): $(LIB)
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

$(LIB): cunn/lib/THCUNN/build/libTHCUNN.a $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(SHARED) $(CFLAGS) $(CCFLAGS) $(OBJECTS) $(CUDA_OBJECTS) $(LINKS) -o $(LIB)

%.o: ../torch/%.c $(HEADERS) $(JAVAH)
	$(NVCC) -x cu $(CFLAGS) $(NVCC_FLAGS) $(GEN_CODES) $(INCLUDES) -c $<

%.o: %.cu $(CUDA_HEADERS) $(JAVAH)
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $(GEN_CODES) $(INCLUDES) -c $<

cutorch/lib/THC/build/libTHC.a:
	cmake -Hcutorch/lib/THC -Bcutorch/lib/THC/build $(CMAKE_FLAGS) -DCMAKE_MODULE_PATH=$(subst $(space),\ ,$(abspath ../cmake)) -DBUILD_STATIC=1 -Wno-dev -DCMAKE_VERBOSE_MAKEFILE=1
	$(MAKE) -C cutorch/lib/THC/build

cunn/lib/THCUNN/build/libTHCUNN.a: cutorch/lib/THC/build/libTHC.a
	cmake -Hcunn/lib/THCUNN -Bcunn/lib/THCUNN/build $(CMAKE_FLAGS) -DCMAKE_MODULE_PATH=$(subst $(space),\ ,$(abspath ../cmake)) -DBUILD_STATIC=1 -DCUDA=1 -DCMAKE_VERBOSE_MAKEFILE=1
	$(MAKE) -C cunn/lib/THCUNN/build

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(CUDA_OBJECTS) $(LIB) $(DEPLOY_DIR)/$(LIB)

.PHONY: cleanall
cleanall: clean
	rm -rf cutorch/lib/THC/build	
	rm -rf cunn/lib/THCUNN/build	
	
