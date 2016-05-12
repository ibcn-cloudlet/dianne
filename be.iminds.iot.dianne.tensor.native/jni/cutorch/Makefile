include ../Makefile

CUDA_HOME := /usr/local/cuda

SRC := ../torch/TensorLoader.c ../torch/Tensor.c ../torch/TensorOps.c ../torch/ModuleOps.c
OBJECTS := TensorLoader.o Tensor.o TensorOps.o THCudaTensorOps.o ModuleOps.o

LINKS := -L. -lTHCUNN -lTHC -L../torch -lTH -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas

INCLUDES += -I.
INCLUDES += -I$(CUDA_HOME)/include/
INCLUDES += -I../torch/torch7/lib/TH -I../torch/torch7/lib/TH/build
INCLUDES += -I../torch/nn/lib/THNN 
INCLUDES += -Icutorch/lib -Icutorch/lib/THC -Icutorch/lib/THC/build
INCLUDES += -Icunn/lib/THNN  

FLAGS += -DCUDA 

NVCC=nvcc -ccbin clang
NVCC_FLAGS := -Xcompiler -fPIC
CUDA_SRC := THCudaTensorOps.cu
CUDA_OBJECTS := $(SRC:.cu=.o)

compile: javah $(SRC)
	$(CC) $(FLAGS) $(CCFLAGS) $(INCLUDES) -c $(SRC)

THCudaTensorOps.o: $(CUDA_SRC)
	$(NVCC) $(FLAGS) $(NVCC_FLAGS) $(INCLUDES) -c $(CUDA_SRC)

link: libTHC.a libTHCUNN.a THCudaTensorOps.o compile 
	$(CC) -shared $(FLAGS) $(CCFLAGS) $(OBJECTS) $(LINKS) -o $(LIB)
	
install: link 
	mkdir -p $(DEPLOY_DIR); \
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

libTHC.a:
	cmake -Hcutorch/lib/THC -Bcutorch/lib/THC/build
	make -C cutorch/lib/THC/build ; \
	cp cutorch/lib/THC/build/libTHC.a libTHC.a

libTHCUNN.a:
	cmake -Hcunn/lib/THCUNN -Bcunn/lib/THCUNN/build
	make -C cunn/lib/THCUNN/build ; \
	cp cunn/lib/THCUNN/build/libTHCUNN.a libTHCUNN.a

clean:
	rm -rf *.o $(LIB) be_*.h test
	
cleanAll: clean
	rm -rf *.a 
	rm -rf cutorch/lib/THC/build	
	rm -rf cunn/lib/THCUNN/build	
	