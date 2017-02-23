include ../Makefile.system

HEADERS := $(wildcard *.h)
SRC := $(wildcard *.c)
OBJECTS=$(SRC:.c=.o)
JAVAH := ../../../be.iminds.iot.dianne.tensor/generated/jni-headers
LINKS := -Lnn/lib/THNN/build -lTHNN -Ltorch7/lib/TH/build -lTH -LOpenBLAS -lopenblas -lgomp -lgfortran -pthread
INCLUDES += -Itorch7/lib/TH -Itorch7/lib/TH/build -Inn/lib/THNN -Inn/lib/THNN/build -I$(JAVAH)

.PHONY: all
all: $(LIB)

.PHONY: info install
install: info $(DEPLOY_DIR)/$(LIB)

info:
	$(info DEPLOY_DIR: $(DEPLOY_DIR)/)

$(DEPLOY_DIR)/$(LIB): $(LIB)
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

$(LIB): OpenBLAS/libopenblas.a torch7/lib/TH/build/libTH.a nn/lib/THNN/build/libTHNN.a $(OBJECTS)
	$(CXX) $(SHARED) $(CFLAGS) $(CCFLAGS) $(OBJECTS) $(LINKS) -o $(LIB)

%.o: %.c $(HEADERS) $(JAVAH)
	$(CXX) $(CFLAGS) $(CCFLAGS) $(INCLUDES) -c $< 

torch7/lib/TH/build/libTH.a: OpenBLAS/libopenblas.a
	cmake -Htorch7/lib/TH -Btorch7/lib/TH/build -DWITH_BLAS=open -DCMAKE_LIBRARY_PATH=$(subst $(space),\ ,$(abspath OpenBLAS)) -DBUILD_STATIC=1
	$(MAKE) -C torch7/lib/TH/build

nn/lib/THNN/build/libTHNN.a: torch7/lib/TH/build/libTH.a
	cmake -Hnn/lib/THNN -Bnn/lib/THNN/build -DCMAKE_MODULE_PATH=$(subst $(space),\ ,$(abspath ../cmake)) -DBUILD_STATIC=1
	$(MAKE) -C nn/lib/THNN/build

OpenBLAS/libopenblas.a:
	# CBLAS and LAPACKE or C wrappers for BLAS and LAPACK. Torch does not use these wrappers so they should not be compiled (saves build time)
	# We do not use the shared library
	$(MAKE) -C OpenBLAS NO_AFFINITY=1 USE_OPENMP=1 NO_SHARED=1 NO_CBLAS=1 NO_LAPACKE=1 NUM_THREADS=64

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(LIB) $(DEPLOY_DIR)/$(LIB)

.PHONY: cleanall
cleanall: clean
	rm -rf torch7/lib/TH/build
	rm -rf nn/lib/THNN/build
	$(MAKE) -C OpenBLAS clean
	
