#!/bin/bash
ARCH=`uname -m`
OS=`uname -s`
NATIVE=native/$OS/$ARCH/

if [ $# -ne 0 ]; then
	if [ $1 == "cuda" ]; then
		NATIVE=cuda/$OS/$ARCH/
	fi
fi

ant compileTests
java -Djava.library.path=$NATIVE -cp bin:bin_test:/usr/share/java/junit4.jar org.junit.runner.JUnitCore be.iminds.iot.dianne.tensor.TensorTest
java -Djava.library.path=$NATIVE -cp bin:bin_test:/usr/share/java/junit4.jar org.junit.runner.JUnitCore be.iminds.iot.dianne.tensor.TensorMathTest

