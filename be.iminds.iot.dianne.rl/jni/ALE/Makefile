# This file builds the native Arcade Learning Environment and the JNI interface
OS ?= $(shell uname -s)
ARCH ?= $(shell uname -m)

ifeq ($(OS),Linux)
    include Makefile.Linux
endif
ifeq ($(OS),Darwin)
    include Makefile.Darwin
    OS := MacOSX
endif

JAVA_INCLUDE := $(JAVA_HOME)/include
JAVA_PLATFORMINCLUDE := $(JAVA_INCLUDE)/$(JDKPLATFORMINCLUDESUBDIR)

JAVAH := ../../generated/jni-headers
SRC := $(wildcard *.cpp)  
INCLUDES := -I$(JAVA_INCLUDE) -I$(JAVA_PLATFORMINCLUDE) -IArcade-Learning-Environment/src -I$(JAVAH)
OBJECTS := $(SRC:.cpp=.o)
FLAGS := -O3 -w
LINKS := -LArcade-Learning-Environment/build -lale -lz
DEPLOY_DIR := ../../native/$(OS)/$(ARCH)
export

.PHONY: install
install: $(DEPLOY_DIR)/$(LIB)

$(DEPLOY_DIR)/$(LIB): $(LIB)
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

$(LIB): Arcade-Learning-Environment/build/libale.a $(OBJECTS)
	$(CXX) $(FLAGS) $(CCFLAGS) $(SHARED) $(OBJECTS) $(LINKS) -o $(LIB)
	
%.o: %.cpp $(JAVAH)
	$(CXX) $(FLAGS) $(CCFLAGS) -fPIC $(INCLUDES) -c $<	

Arcade-Learning-Environment/build/libale.a:
	cmake -HArcade-Learning-Environment -BArcade-Learning-Environment/build -DUSE_SDL=OFF -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF
	$(MAKE) -j 4 -C Arcade-Learning-Environment/build

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(LIB) $(DEPLOY_DIR)/$(LIB)

.PHONY: cleanall
cleanall: clean
	rm -rf Arcade-Learning-Environment/build
	