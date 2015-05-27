# This file builds the native TH Tensor library and the JNI interface
 
CC=gcc
CXX=g++
NVCC=nvcc

JAVA_HOME := /usr/lib/jvm/default-java
CUDA_HOME := /usr/local/cuda

ARCH := $(shell uname -m)
JAVAH := be.iminds.iot.dianne.tensor.impl.th.THTensor be.iminds.iot.dianne.tensor.impl.th.THTensorMath be.iminds.iot.dianne.tensor.impl.th.THTensorFactory
SRC := THTensorFactoryJNI.c THTensorJNI.c THTensorMathJNI.c 
INCLUDES := -I$(JAVA_HOME)/include/ -I/usr/lib/jvm -ITH
OBJECT := THTensorFactoryJNI.o THTensorJNI.o THTensorMathJNI.o 
FLAGS := -O3 -w
LINKS := libTH.a libopenblas.a -lm -lgomp
LIB := libTHTensor.so
DEPLOY_DIR := ../../native/$(ARCH)

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
cuda: DEPLOY_DIR := ../../native/$(ARCH)/cuda
cuda: cudaops libTHC.a install	

javah:
	for file in $(JAVAH) ; do \
    	javah -jni -classpath ../../bin $$file   ; \
	done
	
compile: javah
	$(CC) $(FLAGS) -fPIC -DLINUX $(INCLUDES) -c $(SRC)

cudaops:
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(CUDA_SRC)

link: compile 
	$(CC) $(FLAGS) -shared $(OBJECT) $(LINKS) -o $(LIB) 
	
install: libopenblas.a libTH.a link 
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

test:
	$(CXX) $(FLAGS) -DLINUX $(INCLUDES) -c Test.c
	$(CXX) $(FLAGS) Test.o $(LINKS) -o test

libTH.a:
	cd TH; mkdir -p build; cd build; rm -f libTH.a; cmake -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} ..; make; cp libTH.a ../..; cp THGeneral.h ..

libTHC.a:
	cd THC; mkdir -p build; cd build; rm -f libTHC.a; cmake -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} ..; make; cp libTHC.a ../..

libopenblas.a:
	cd OpenBLAS; make; cp libopenblas.a ..
	
clean:
	rm -rf *.o ${LIB} be_*.h test
	
cleanAll: clean
	rm -rf TH/build libTH.a; rm -rf THC/build libTHC.a; rm libopenblas.a; cd OpenBLAS; make clean

