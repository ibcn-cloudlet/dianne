include ../Makefile.system

HEADERS := $(wildcard *.h)
SRC := $(wildcard *.c)
SRC_TORCH := $(wildcard ../torch/*.c)
OBJECTS_TORCH := $(SRC_TORCH:.c=.o)
OBJECTS=$(SRC:.c=.o)
JAVAH := ../../../be.iminds.iot.dianne.tensor/generated/jni-headers
LINKS := -L../torch/nn/lib/THNN/build -lTHNN -L../torch/torch7/lib/TH/build -lTH -LOpenBLAS -lopenblas -lgomp -lgfortran -pthread
INCLUDES += -I../torch/torch7/lib/TH -I../torch/torch7/lib/TH/build -I../torch -I../torch/nn/lib/THNN -I../torch/nn/lib/THNN/build -I$(JAVAH)

.PHONY: all
all: $(LIB)

.PHONY: info install
install: info $(DEPLOY_DIR)/$(LIB)

info:
	$(info DEPLOY_DIR: $(DEPLOY_DIR)/)
	
$(DEPLOY_DIR)/$(LIB): $(LIB)
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)	

$(LIB): symbolcleanup
	$(CXX) $(SHARED) $(CFLAGS) $(CCFLAGS) $(OBJECTS) $(OBJECTS_TORCH) $(LINKS) -o $(LIB)

## erase all Java_... methods implemented in CUDNNModuleOps from the torch ModuleOps
symbolcleanup: $(OBJECTS) $(OBJECTS_TORCH)
	nm CUDNNModuleOps.o | grep -o "Java_.*" > symbols
	objcopy --strip-symbols=symbols ../torch/ModuleOps.o ../torch/ModuleOps.o

$(OBJECTS_TORCH):
	$(MAKE) -C ../torch
	
%.o: %.c $(HEADERS) $(JAVAH)
	$(CXX) $(CFLAGS) $(CCFLAGS) $(INCLUDES) -c $<
	
.PHONY: clean
clean:
	rm -f symbols
	rm -f $(OBJECTS) $(LIB) 	