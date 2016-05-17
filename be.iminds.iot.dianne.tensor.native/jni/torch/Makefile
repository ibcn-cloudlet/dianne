include ../Makefile

SRC := TensorLoader.c Tensor.c TensorOps.c ModuleOps.c
LINKS := -L. -lTHNN -lTH -lopenblas -lgomp 
INCLUDES += -Itorch7/lib/TH -Itorch7/lib/TH/build
INCLUDES += -Inn/lib/THNN 

compile: javah $(SRC)
	$(CC) $(FLAGS) $(CCFLAGS) $(INCLUDES) -c $(SRC)

link: libopenblas.a libTH.a libTHNN.a compile 
	$(CC) -shared $(FLAGS) $(CCFLAGS) $(OBJECTS) $(LINKS) -o $(LIB)
	
install: link 
	mkdir -p $(DEPLOY_DIR); \
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

libTH.a:
	cmake -Htorch7/lib/TH -Btorch7/lib/TH/build
	make -C torch7/lib/TH/build ; \
	cp torch7/lib/TH/build/libTH.a libTH.a

libTHNN.a:
	cmake -Hnn/lib/THNN -Bnn/lib/THNN/build
	make -C nn/lib/THNN/build ; \
	cp nn/lib/THNN/build/libTHNN.a libTHNN.a

libopenblas.a:
	cd OpenBLAS; make USE_OPENMP=1; cp libopenblas.a ..

clean:
	rm -rf *.o $(LIB) be_*.h test
	
cleanAll: clean
	rm -rf *.a 
	rm -rf torch7/lib/TH/build	
	rm -rf nn/lib/THNN/build	
	cd OpenBLAS; make clean
	