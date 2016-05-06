include ../Makefile

SRC := TensorLoader.c Tensor.c TensorOps.c ModuleOps.c
LINKS := -L. -lTHNN -lTH -lopenblas -lgomp 
INCLUDES += -Itorch7/lib/TH -Itorch7/lib/TH/build
INCLUDES += -Inn/lib/THNN 



compile: javah $(SRC)
	$(CC) $(FLAGS) $(CCFLAGS) $(INCLUDES) -c $(SRC)

link: libTH.a libTHNN.a compile 
	$(CC) -shared $(FLAGS) $(CCFLAGS) $(OBJECTS) $(LINKS) -o $(LIB)
	
install: link 
	mkdir -p $(DEPLOY_DIR); \
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

libTH.a:
	cmake -Htorch7/lib/TH -Btorch7/lib/TH/build -DBUILD_STATIC=TRUE -DCMAKE_INSTALL_PREFIX=. -DCMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE
	make -C torch7/lib/TH/build ; \
	cp torch7/lib/TH/build/libTH_static.a libTH.a

libTHNN.a:
	cmake -Hnn/lib/THNN -Bnn/lib/THNN/build -DBUILD_STATIC=TRUE
	make -C nn/lib/THNN/build ; \
	cp nn/lib/THNN/build/libTHNN_static.a libTHNN.a

clean:
	rm -rf *.o $(LIB) be_*.h test
	
cleanAll: clean
	rm -rf *.a 
	rm -rf torch7/lib/TH/build	
	rm -rf nn/lib/THNN/build	
	