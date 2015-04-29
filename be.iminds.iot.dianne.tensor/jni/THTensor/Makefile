# This file builds the native TH Tensor library and the JNI interface
 
CC=gcc
CXX=g++

JAVAH := be.iminds.iot.dianne.tensor.impl.th.THTensor be.iminds.iot.dianne.tensor.impl.th.THTensorMath
SRC := THTensorJNI.c THTensorMathJNI.c 
INCLUDES := -I/usr/lib/jvm/jdk1.8.0_25/include/ -I/usr/lib/jvm
OBJECT := THTensorJNI.o THTensorMathJNI.o 
FLAGS := -O3
LINKS := libTH.a
LIB := libTHTensor.so
ARCH := x86_64

all: install  

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

buildTH:
	cd TH; mkdir -p build; cd build; rm -f libTH.a; cmake -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC} ..; make; cp libTH.a ../..; cp THGeneral.h ..

cleanTH:
	rm -rf TH/build libTH.a 

clean:
	rm -rf *.o ${LIB} be_*.h 
