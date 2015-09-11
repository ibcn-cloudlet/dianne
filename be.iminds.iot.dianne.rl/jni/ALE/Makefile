# This file builds the native Arcade Learning Environment and the JNI interface
OS := $(shell uname -s)
ARCH := $(shell uname -m)

ifeq ($(OS),Linux)
    include Makefile.Linux
endif
ifeq ($(OS),Darwin)
    include Makefile.Darwin
endif

JAVA_INCLUDE := $(JAVA_HOME)/include
JAVA_PLATFORMINCLUDE := $(JAVA_INCLUDE)/$(JDKPLATFORMINCLUDESUBDIR)

JAVAH := be.iminds.iot.dianne.rl.ale.ArcadeLearningEnvironment
SRC := ALE.cpp  
INCLUDES := -I$(JAVA_INCLUDE) -I$(JAVA_PLATFORMINCLUDE) -IArcade-Learning-Environment/src
OBJECT := ALE.o 
FLAGS := -O3 -w
LINKS := libale.a -lz
DEPLOY_DIR := ../../native/$(OS)/$(ARCH)

all: install 

# the generated header file is already put in the git repository
javah:
	for file in $(JAVAH); do \
    	javah -jni -classpath ../../bin:../../../be.iminds.iot.dianne.tensor/generated/be.iminds.iot.dianne.tensor.api.jar $$file; \
	done
	
compile:
	$(CXX) $(FLAGS) $(CCFLAGS) -fPIC $(INCLUDES) -c $(SRC)
	
link: compile 
	$(CXX) $(FLAGS) $(CCFLAGS) $(SHARED) $(OBJECT) $(LINKS) -o $(LIB) 
	
install: libale.a link 
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)	


libale.a:
	cd Arcade-Learning-Environment; cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF; make -j 4; cp libale.a ..
	
clean:
	rm -rf *.o $(LIB)

cleanAll: clean
	rm libale.a; cd Arcade-Learning-Environment; make clean
	