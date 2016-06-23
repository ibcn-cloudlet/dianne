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
$(info CUDA_TARGET: $(CUDA_TARGET))

ifneq (,$(filter $(ARCH),x86_64 aarch64))
    TARGET_SIZE := 64
else ifeq ($(findstring arm,$(ARCH)),arm)
    TARGET_SIZE := 32
else
    $(error ERROR - unsupported value $(ARCH) for ARCH!)
endif

SRC := $(wildcard ../torch/*.c)
OBJECTS := $(notdir $(SRC:.c=.o))
JAVAH := ../../../be.iminds.iot.dianne.tensor/generated/jni-headers
LINKS := -Lcunn/lib/THCUNN/build -lTHCUNN -Lcutorch/lib/THC/build -lTHC -L../torch/torch7/lib/TH/build -lTH -L$(CUDA_TARGET)/lib -lcudart_static -lcublas_static -lstdc++ -lculibos -pthread
INCLUDES += -I.
INCLUDES += -I$(CUDA_TARGET)/include/
INCLUDES += -Icutorch/lib/THC -Icutorch/lib/THC/build
INCLUDES += -Icunn/lib/THCUNN -Icunn/lib/THCUNN/build
INCLUDES += -I../torch/torch7/lib/TH -I../torch/torch7/lib/TH/build
INCLUDES += -I../torch/nn/lib/THNN -I../torch/nn/lib/THNN/build
INCLUDES += -I$(JAVAH)
CCFLAGS += -DCUDA 

NVCC=nvcc -ccbin $(CXX)
export NVCC
NVCC_FLAGS := -DCUDA -Xcompiler -fPIC -m${TARGET_SIZE}
CUDA_SRC := $(wildcard *.cu)
CUDA_OBJECTS := $(CUDA_SRC:.cu=.o)

ifneq ($(strip $(OS))-$(strip $(ARCH)),$(uname -s)-$(uname -m))
	CMAKE_FLAGS += -DCMAKE_SYSTEM_NAME=$(OS) -DCMAKE_SYSTEM_PROCESSOR=$(ARCH)
endif

.PHONY: install
install: $(DEPLOY_DIR)/$(LIB)

$(DEPLOY_DIR)/$(LIB): $(LIB)
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

$(LIB): cunn/lib/THCUNN/build/libTHCUNN.a $(OBJECTS) $(CUDA_OBJECTS)
	$(CC) $(SHARED) $(CFLAGS) $(CCFLAGS) $(OBJECTS) $(CUDA_OBJECTS) $(LINKS) -o $(LIB)

%.o: ../torch/%.c $(JAVAH)
	$(CC) $(CFLAGS) $(CCFLAGS) $(INCLUDES) -c $<

%.o: %.cu $(JAVAH)
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $(INCLUDES) -c $<

cutorch/lib/THC/build/libTHC.a:
	cmake -Hcutorch/lib/THC -Bcutorch/lib/THC/build $(CMAKE_FLAGS) -DCMAKE_MODULE_PATH="$(abspath ../cmake)" -DBUILD_STATIC=1 -Wno-dev
	$(MAKE) -C cutorch/lib/THC/build

cunn/lib/THCUNN/build/libTHCUNN.a: cutorch/lib/THC/build/libTHC.a
	cmake -Hcunn/lib/THCUNN -Bcunn/lib/THCUNN/build $(CMAKE_FLAGS) -DCMAKE_MODULE_PATH="$(abspath ../cmake)" -DBUILD_STATIC=1 -DCUDA=1
	$(MAKE) -C cunn/lib/THCUNN/build

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(CUDA_OBJECTS) $(LIB) $(DEPLOY_DIR)/$(LIB)

.PHONY: cleanall
cleanall: clean
	rm -rf cutorch/lib/THC/build	
	rm -rf cunn/lib/THCUNN/build	
	
