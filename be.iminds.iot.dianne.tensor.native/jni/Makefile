# This file builds the native TH Tensor library and the JNI interface
OS := $(shell uname -s)
ARCH := $(shell uname -m)
GPU := $(shell lspci -v 2>/dev/null | grep NVIDIA)

# Java config
JAVA_HOME ?= /usr/lib/jvm/default-java
# this is the subdirectory name in JAVA_HOME/include/ that contains platform-specific files
# it seems that some versions of Java 1.1 on linux need this to be "genunix"
JDKPLATFORMINCLUDESUBDIR := linux
JAVA_INCLUDE := $(JAVA_HOME)/include
JAVA_PLATFORMINCLUDE := $(JAVA_INCLUDE)/$(JDKPLATFORMINCLUDESUBDIR)
JAVAH := be.iminds.iot.dianne.tensor.Tensor be.iminds.iot.dianne.tensor.TensorOps

# Compiler config
CC=gcc
CXX=g++
FLAGS += -DLINUX
CCFLAGS := -O3 -w -fPIC
LIB := libTensor.so

INCLUDES := -I$(JAVA_INCLUDE) -I$(JAVA_PLATFORMINCLUDE)
OBJECTS=$(SRC:.c=.o)
DEPLOY_DIR := ../../native/$(OS)/$(ARCH)


all: install  

javah: 
	for file in $(JAVAH); do \
    	  javah -jni -classpath ../../../be.iminds.iot.dianne.tensor/bin $$file; \
	done
	
compile: javah $(SRC)
	$(CC) $(FLAGS) $(CCFLAGS) $(INCLUDES) -c $(SRC)

link: compile 
	$(CC) -shared $(FLAGS) $(CCFLAGS) $(OBJECTS) $(LINKS) -o $(LIB)
	
install: link 
	mkdir -p $(DEPLOY_DIR); \
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

clean:
	rm -rf *.o $(LIB) be_*.h test
	
