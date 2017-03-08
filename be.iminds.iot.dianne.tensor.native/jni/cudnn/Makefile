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

HEADERS := $(wildcard *.h)
SRC := $(wildcard *.c)
OBJECTS=$(SRC:.c=.o)
SRC_TORCH := $(wildcard ../torch/*.c)
SRC_CUTORCH := $(wildcard ../cutorch/*.cu)
OBJECTS_CUTORCH := $(SRC_CUTORCH:.cu=.o)
OBJECTS_CUTORCH += $(addprefix ../cutorch/, $(notdir $(SRC_TORCH:.c=.o)))
JAVAH := ../../../be.iminds.iot.dianne.tensor/generated/jni-headers
LINKS := -L../cutorch/cunn/lib/THCUNN/build -lTHCUNN -L../cutorch/cutorch/lib/THC/build -lTHC -L../torch/torch7/lib/TH/build -lTH -L$(CUDA_TARGET)/$(CUDA_LIB) -lcudnn_static -lcudart_static -lcublas_static -lstdc++ -lculibos -pthread
INCLUDES += -I$(CUDA_TARGET)/include/
INCLUDES += -I../cutorch/cutorch/lib -I../cutorch/cutorch/lib/THC -I../cutorch/cutorch/lib/THC/build
INCLUDES += -I../cutorch/cunn/lib/THCUNN -I../cutorch/cunn/lib/THCUNN/build
INCLUDES += -I../torch/torch7/lib/TH -I../torch/torch7/lib/TH/build -I../torch 
INCLUDES += -I$(JAVAH)

NVCC=nvcc -ccbin $(CXX)
export NVCC
NVCC_FLAGS := -DCUDA -Xcompiler -fPIC -m${TARGET_SIZE}

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

.PHONY: all
all: $(LIB)

.PHONY: info install
install: info $(DEPLOY_DIR)/$(LIB)

info:
	$(info DEPLOY_DIR: $(DEPLOY_DIR)/)
	
$(DEPLOY_DIR)/$(LIB): $(LIB)
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)	

$(LIB): symbolcleanup $(OBJECTS) $(OBJECTS_CUTORCH)
	$(CXX) $(SHARED) $(CFLAGS) $(CCFLAGS) $(OBJECTS) $(OBJECTS_CUTORCH) $(LINKS) -o $(LIB)

## erase all Java_... methods implemented in CUDNNModuleOps from the torch ModuleOps
symbolcleanup: $(OBJECTS) $(OBJECTS_CUTORCH)
	nm CudnnModuleOps.o | grep -o "Java_.*" > symbols
	objcopy --strip-symbols=symbols ../cutorch/ModuleOps.o ../cutorch/ModuleOps.o
	nm CudnnTensorLoader.o | grep -o "Java_.*" > symbols
	objcopy --strip-symbols=symbols ../cutorch/TensorLoader.o ../cutorch/TensorLoader.o
	
%.o: %.c $(HEADERS) $(JAVAH)
	$(NVCC) -x cu $(CFLAGS) $(NVCC_FLAGS) $(GEN_CODES) $(INCLUDES) -c $<
	
.PHONY: clean
clean:
	rm -f symbols
	rm -f $(OBJECTS) $(LIB) $(DEPLOY_DIR)/$(LIB)
	
.PHONY: cleanall
cleanall: clean
	
