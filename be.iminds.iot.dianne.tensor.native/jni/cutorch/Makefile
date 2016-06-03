CUDA_HOME := /usr/local/cuda

SRC := $(wildcard ../torch/*.c)
OBJECTS := $(notdir $(SRC:.c=.o))
JAVAH := $(abspath ../../../be.iminds.iot.dianne.tensor/generated/jni-headers)
LINKS := -Lcunn/lib/THCUNN/build -lTHCUNN -Lcutorch/lib/THC/build -lTHC -L../torch/torch7/lib/TH/build -lTH -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
INCLUDES += -I.
INCLUDES += -I$(CUDA_HOME)/include/
INCLUDES += -Icutorch/lib/THC -Icutorch/lib/THC/build
INCLUDES += -Icunn/lib/THCUNN -Icunn/lib/THCUNN/build
INCLUDES += -I../torch/torch7/lib/TH -I../torch/torch7/lib/TH/build
INCLUDES += -I../torch/nn/lib/THNN -I../torch/nn/lib/THNN/build
INCLUDES += -I$(JAVAH)
FLAGS += -DCUDA 

NVCC=nvcc -ccbin clang
export NVCC
NVCC_FLAGS := -Xcompiler -fPIC
CUDA_SRC := THCudaTensorOps.cu
CUDA_OBJECTS := $(CUDA_SRC:.cu=.o)

.PHONY: install
install: $(DEPLOY_DIR)/$(LIB)

$(DEPLOY_DIR)/$(LIB): $(LIB)
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

$(LIB): cutorch/lib/THC/build/libTHC.a cunn/lib/THCUNN/build/libTHCUNN.a $(OBJECTS) $(CUDA_OBJECTS)
	$(CC) $(SHARED) $(FLAGS) $(CCFLAGS) $(OBJECTS) $(LINKS) -o $(LIB)

%.o: ../torch/%.c $(JAVAH)
	$(CC) $(FLAGS) $(CCFLAGS) $(INCLUDES) -c $<

%.o: %.cu $(JAVAH)
	$(NVCC) $(FLAGS) $(NVCC_FLAGS) $(INCLUDES) -c $<

cutorch/lib/THC/build/libTHC.a:
	cmake -Hcutorch/lib/THC -Bcutorch/lib/THC/build
	$(MAKE) -C cutorch/lib/THC/build

cunn/lib/THCUNN/build/libTHCUNN.a:
	cmake -Hcunn/lib/THCUNN -Bcunn/lib/THCUNN/build
	$(MAKE) -C cunn/lib/THCUNN/build

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(LIB) $(DEPLOY_DIR)/$(LIB)

.PHONY: cleanall
cleanall: clean
	rm -rf cutorch/lib/THC/build	
	rm -rf cunn/lib/THCUNN/build	
	
