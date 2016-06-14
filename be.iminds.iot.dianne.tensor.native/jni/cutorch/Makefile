CUDA_HOME ?= /usr/local/cuda

SRC := $(wildcard ../torch/*.c)
OBJECTS := $(notdir $(SRC:.c=.o))
JAVAH := ../../../be.iminds.iot.dianne.tensor/generated/jni-headers
LINKS := -Lcunn/lib/THCUNN/build -lTHCUNN -Lcutorch/lib/THC/build -lTHC -L../torch/torch7/lib/TH/build -lTH -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib -lcudart_static -lcublas_static -lstdc++ -lculibos
INCLUDES += -I.
INCLUDES += -I$(CUDA_HOME)/include/
INCLUDES += -Icutorch/lib/THC -Icutorch/lib/THC/build
INCLUDES += -Icunn/lib/THCUNN -Icunn/lib/THCUNN/build
INCLUDES += -I../torch/torch7/lib/TH -I../torch/torch7/lib/TH/build
INCLUDES += -I../torch/nn/lib/THNN -I../torch/nn/lib/THNN/build
INCLUDES += -I$(JAVAH)
CCFLAGS += -DCUDA 

NVCC=nvcc -ccbin $(CXX)
export NVCC
NVCC_FLAGS := -DCUDA -Xcompiler -fPIC
CUDA_SRC := $(wildcard *.cu)
CUDA_OBJECTS := $(CUDA_SRC:.cu=.o)

.PHONY: install
install: $(DEPLOY_DIR)/$(LIB)

$(DEPLOY_DIR)/$(LIB): $(LIB)
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

$(LIB): cutorch/lib/THC/build/libTHC.a cunn/lib/THCUNN/build/libTHCUNN.a $(OBJECTS) $(CUDA_OBJECTS)
	$(CC) $(SHARED) $(CFLAGS) $(CCFLAGS) $(OBJECTS) $(CUDA_OBJECTS) $(LINKS) -o $(LIB)

%.o: ../torch/%.c $(JAVAH)
	$(CC) $(CFLAGS) $(CCFLAGS) $(INCLUDES) -c $<

%.o: %.cu $(JAVAH)
	$(NVCC) $(CFLAGS) $(NVCC_FLAGS) $(INCLUDES) -c $<

cutorch/lib/THC/build/libTHC.a:
	cmake -Hcutorch/lib/THC -Bcutorch/lib/THC/build
	$(MAKE) -C cutorch/lib/THC/build

cunn/lib/THCUNN/build/libTHCUNN.a:
	cmake -Hcunn/lib/THCUNN -Bcunn/lib/THCUNN/build
	$(MAKE) -C cunn/lib/THCUNN/build

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(CUDA_OBJECTS) $(LIB) $(DEPLOY_DIR)/$(LIB)

.PHONY: cleanall
cleanall: clean
	rm -rf cutorch/lib/THC/build	
	rm -rf cunn/lib/THCUNN/build	
	
