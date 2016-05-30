# This file builds the native TH Tensor library and the JNI interface
OS := $(shell uname -s)
ARCH := $(shell uname -m)
GPU := $(shell lspci -v 2>/dev/null | grep NVIDIA)

#Genrate Makefile.conf
DUMMY := $(shell make -C . -f Makefile.prebuild)
include Makefile.conf

ifeq ($(OS),Linux)
    include Makefile.Linux
endif
ifeq ($(OS),Darwin)
    include Makefile.Darwin
endif

export CC
export CXX
export NVCC

CUDA_HOME := /usr/local/cuda
JAVA_INCLUDE := $(JAVA_HOME)/include
JAVA_PLATFORMINCLUDE := $(JAVA_INCLUDE)/$(JDKPLATFORMINCLUDESUBDIR)

JAVAH := be.iminds.iot.dianne.tensor.impl.th.THTensor be.iminds.iot.dianne.tensor.impl.th.THTensorMath be.iminds.iot.dianne.tensor.impl.th.THTensorFactory
SRC := THTensorFactoryJNI.c THTensorJNI.c THTensorMathJNI.c
INCLUDES := -Iinclude/TH -I$(JAVA_INCLUDE) -I$(JAVA_PLATFORMINCLUDE)
OBJECTS=$(SRC:.c=.o)
CCFLAGS := -O3 -w -fPIC
LINKS := -Llib -lTH -lopenblas $(SHARED) -lm -lgomp -pthread
ifndef NOFORTRAN
LINKS += -lgfortran
endif
DEPLOY_DIR := ../../generic/$(OS)/$(ARCH)

# cuda specific build parameters
CUDA_INCLUDES := -ITHC -I$(CUDA_HOME)/include/
CUDA_LINKS := -Llib -lTHC -lTH $(SHARED) -lgomp -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
CUDA_FLAGS := -DCUDA
NVCC_FLAGS := -Xcompiler -fPIC
CUDA_SRC := THCudaTensorOps.cu
CUDA_OBJECTS := $(CUDA_SRC:.cu=.o)

# check if it is the Jetson TK board
ifneq (,$(findstring 0e13,$(GPU)))
    # Found
    FLAGS += -DJETSON
endif

all: install  

cuda: INCLUDES += $(CUDA_INCLUDES)
cuda: LINKS = $(CUDA_LINKS)
cuda: FLAGS += $(CUDA_FLAGS)
cuda: OBJECTS += $(CUDA_OBJECTS)
cuda: DEPLOY_DIR := ../../cuda/$(OS)/$(ARCH)
cuda: cudaops lib/libTHC.a install	

javah: 
	for file in $(JAVAH); do \
    	  javah -jni -classpath ../../bin $$file; \
	done
	
compile: javah $(SRC)
	$(CC) $(FLAGS) $(CCFLAGS) $(INCLUDES) -c $(SRC)

cudaops: $(CUDA_SRC)
	$(NVCC) $(FLAGS) $(NVCC_FLAGS) $(INCLUDES) -c $(CUDA_SRC)

link: compile 
	$(CC) $(FLAGS) $(CCFLAGS) $(OBJECTS) $(LINKS) -o $(LIB)
	
install: lib/libopenblas.a lib/libTH.a link 
	mkdir -p $(DEPLOY_DIR); \
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

lib/libTH.a:
	cmake -HTH -BTH/build -DCMAKE_INSTALL_PREFIX=. -DCMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE
	make -C TH/build ; \
	make -C TH/build install

lib/libTHC.a:
	cmake -Hcutorch/lib/THC -Bcutorch/lib/THC/build -DCMAKE_INSTALL_PREFIX=. ; \
	make -C cutorch/lib/THC/build ; \
	make -C cutorch/lib/THC/build install

lib/libopenblas.a:
	make -C OpenBLAS ; \
	make -C OpenBLAS NO_SHARED=1 PREFIX=.. install
	
clean:
	rm -rf *.o $(LIB) be_*.h test
	
cleanAll: clean
	rm -rf TH/build cutorch/lib/THC/build lib include share bin; \
	make -C OpenBLAS clean
