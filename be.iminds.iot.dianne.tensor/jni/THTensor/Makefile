# This file builds the native TH Tensor library and the JNI interface
OS := $(shell uname -s)
ARCH := $(shell uname -m)

ifeq ($(OS),Linux)
    include Makefile.Linux
endif
ifeq ($(OS),Darwin)
    include Makefile.Darwin
endif

CUDA_HOME := /usr/local/cuda
JAVA_INCLUDE := $(JAVA_HOME)/include
JAVA_PLATFORMINCLUDE := $(JAVA_INCLUDE)/$(JDKPLATFORMINCLUDESUBDIR)

JAVAH := be.iminds.iot.dianne.tensor.impl.th.THTensor be.iminds.iot.dianne.tensor.impl.th.THTensorMath be.iminds.iot.dianne.tensor.impl.th.THTensorFactory
SRC := THTensorFactoryJNI.c THTensorJNI.c THTensorMathJNI.c 
INCLUDES := -I$(JAVA_INCLUDE) -I$(JAVA_PLATFORMINCLUDE) -ITH
OBJECT := THTensorFactoryJNI.o THTensorJNI.o THTensorMathJNI.o 
FLAGS := -O3 -w
LINKS := libTH.a libopenblas.a -lm -lgomp
DEPLOY_DIR := ../../native/$(OS)/$(ARCH)

# cuda specific build parameters
CUDA_INCLUDES := -ITHC -I$(CUDA_HOME)/include/
CUDA_LINKS := libTHC.a libTH.a -lgomp -L$(CUDA_HOME)/lib -lcudart -lcublas
CUDA_FLAGS := -DCUDA
NVCC_FLAGS := -Xcompiler -fPIC
CUDA_SRC := THCudaTensorOps.cu
CUDA_OBJECT := THCudaTensorOps.o


all: install  

cuda: INCLUDES += $(CUDA_INCLUDES)
cuda: LINKS = $(CUDA_LINKS)
cuda: FLAGS += $(CUDA_FLAGS)
cuda: OBJECT += $(CUDA_OBJECT)
cuda: DEPLOY_DIR := ../../cuda/$(OS)/$(ARCH)
cuda: cudaops libTHC.a install	

javah:
	for file in $(JAVAH); do \
    	javah -jni -classpath ../../bin $$file; \
	done
	
compile: javah
	$(CC) $(FLAGS) $(CCFLAGS) -fPIC $(INCLUDES) -c $(SRC)

cudaops:
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(CUDA_SRC)

link: compile 
	$(CC) $(FLAGS) $(CCFLAGS) $(SHARED) $(OBJECT) $(LINKS) -o $(LIB) 
	
install: libopenblas.a libTH.a link 
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

test:
	$(CXX) $(FLAGS) $(INCLUDES) -c Test.c
	$(CXX) $(FLAGS) Test.o $(LINKS) -o test

libTH.a:
	cd TH; mkdir -p build; cd build; rm -f libTH.a; cmake ..; make; cp libTH.a ../..; cp THGeneral.h ..

libTHC.a:
	cd THC; mkdir -p build; cd build; rm -f libTHC.a; cmake ..; make; cp libTHC.a ../..

libopenblas.a:
	cd OpenBLAS; make; cp libopenblas.a ..
	
clean:
	rm -rf *.o $(LIB) be_*.h test
	
cleanAll: clean
	rm -rf THGeneral.h TH/build libTH.a; rm -rf THC/build libTHC.a; rm libopenblas.a; cd OpenBLAS; make clean

