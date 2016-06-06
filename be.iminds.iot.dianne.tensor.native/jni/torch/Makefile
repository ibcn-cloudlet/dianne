SRC := $(wildcard *.c)
OBJECTS=$(SRC:.c=.o)
JAVAH := ../../../be.iminds.iot.dianne.tensor/generated/jni-headers
LINKS := -Lnn/lib/THNN/build -lTHNN -Ltorch7/lib/TH/build -lTH -LOpenBLAS -lopenblas -lgomp -lgfortran
INCLUDES += -Itorch7/lib/TH -Itorch7/lib/TH/build -Inn/lib/THNN -Inn/lib/THNN/build -I$(JAVAH)

.PHONY: install
install: $(DEPLOY_DIR)/$(LIB)

$(DEPLOY_DIR)/$(LIB): $(LIB)
	mkdir -p $(DEPLOY_DIR)
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

$(LIB): OpenBLAS/libopenblas.a torch7/lib/TH/build/libTH.a nn/lib/THNN/build/libTHNN.a $(OBJECTS)
	$(CC) $(SHARED) $(FLAGS) $(CCFLAGS) $(OBJECTS) $(LINKS) -o $(LIB)

%.o: %.c $(JAVAH)
	$(CC) $(FLAGS) $(CCFLAGS) $(INCLUDES) -c $< 

torch7/lib/TH/build/libTH.a:
	cmake -Htorch7/lib/TH -Btorch7/lib/TH/build -DCMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE
	$(MAKE) -C torch7/lib/TH/build

nn/lib/THNN/build/libTHNN.a:
	cmake -Hnn/lib/THNN -Bnn/lib/THNN/build
	$(MAKE) -C nn/lib/THNN/build

OpenBLAS/libopenblas.a:
	$(MAKE) -C OpenBLAS USE_OPENMP=1 NO_SHARED=1

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(LIB) $(DEPLOY_DIR)/$(LIB)

.PHONY: cleanAll
cleanall: clean
	rm -rf torch7/lib/TH/build
	rm -rf nn/lib/THNN/build
	$(MAKE) -C OpenBLAS clean
	
