# This file builds the native TH Tensor library and the JNI interface
 
CC=gcc
CXX=g++

JAVA_HOME := /usr/lib/jvm/jdk1.8.0_25

JAVAH := be.iminds.iot.dianne.tensor.impl.th.THTensor be.iminds.iot.dianne.tensor.impl.th.THTensorMath
SRC := THTensorJNI.c THTensorMathJNI.c 
INCLUDES := -I$(JAVA_HOME)/include/ -I/usr/lib/jvm -ITH
OBJECT := THTensorJNI.o THTensorMathJNI.o 
FLAGS := -O3 -w
LINKS := libTH.a -lm -lgomp -lopenblas
LIB := libTHTensor.so

# cuda specific build parameters
CUDA_INCLUDES := -ITHC -I/usr/local/cuda/include/
CUDA_LINKS := libTHC.a
CUDA_FLAGS := -DCUDA

ARCH := $(shell uname -m)

all: install  

cuda: INCLUDES += $(CUDA_INCLUDES)
cuda: LINKS += $(CUDA_LINKS)
cuda: FLAGS += $(CUDA_FLAGS)
cuda: ARCH += "/cuda/"
cuda: install	

javah:
	for file in $(JAVAH) ; do \
    	javah -jni -classpath ../../bin $$file   ; \
	done
	
compile: javah
	$(CC) $(FLAGS) -fPIC -DLINUX $(INCLUDES) -c $(SRC)

link: compile 
	$(CC) $(FLAGS) -shared $(OBJECT) $(LINKS) -o $(LIB) 
	
install: link
	mkdir -p ../../native/$(ARCH)
	cp $(LIB) ../../native/$(ARCH)/$(LIB)

test: #cuda
	$(CC) $(FLAGS) -DLINUX $(INCLUDES) -c Test.c
	$(CC) $(FLAGS) Test.o $(LINKS) -o test

buildTH:
	cd TH; mkdir -p build; cd build; rm -f libTH.a; cmake -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} ..; make; cp libTH.a ../..; cp THGeneral.h ..

cleanTH:
	rm -rf TH/build libTH.a 

buildTHC:
	cd THC; mkdir -p build; cd build; rm -f libTHC.a; cmake -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} ..; make; cp libTHC.a ../..

cleanTHC:
	rm -rf THC/build libTHC.a 

clean:
	rm -rf *.o ${LIB} be_*.h test
