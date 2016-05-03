include ../Makefile

SRC := Tensor.c TensorOps.c
LINKS := -lTH


compile: javah $(SRC)
	$(CC) $(FLAGS) $(CCFLAGS) $(INCLUDES) -c $(SRC)

link: libTH.a compile 
	$(CC) -shared $(FLAGS) $(CCFLAGS) $(OBJECTS) $(LINKS) -o $(LIB)
	
install: link 
	mkdir -p $(DEPLOY_DIR); \
	cp $(LIB) $(DEPLOY_DIR)/$(LIB)

libTH.a:
	cmake -Htorch7/lib/TH -Btorch7/lib/TH/build -DBUILD_STATIC=TRUE -DCMAKE_INSTALL_PREFIX=. -DCMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE
	make -C torch7/lib/TH/build ; \
	cp torch7/lib/TH/build/libTH_static.a libTH.a

clean:
	rm -rf *.o $(LIB) be_*.h test
	
cleanAll: clean
	rm -rf *.a 
	rm -rf torch7/lib/TH/build	
	
	