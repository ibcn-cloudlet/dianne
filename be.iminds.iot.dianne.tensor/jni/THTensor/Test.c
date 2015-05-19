
#include <stdio.h>

#ifdef CUDA
#include "THCudaTensorJNI.h"
#else
#include "THTensorJNI.h"
#endif

/**
 * This is a sample test program for testing compilation and linking of both Cuda/Float tensors
 */
int main(){
	printf("Tensor: \n");

	THCState* state = (THCState*)malloc(sizeof(THCState));
	THCudaInit(state);

	printf("Initialized CUDA %d %d \n", state->numDevices, state->numUserStreams);

//	THTensor* tensor = THTensor_(newWithSize2d)(2, 2);
//	THTensor_(fill)(tensor, 1.0f);
//	real* data = THTensor_(data)(tensor);
//
//	printf(" %f %f \n %f %f \n", data[0], data[1], data[2], data[3]);
//
//	THTensor_(add)(tensor, tensor, 1.0f);
//
//	printf("Add 1: \n");
//
//	data = THTensor_(data)(tensor);
//	printf(" %f %f \n %f %f \n", data[0], data[1], data[2], data[3]);
//
//	THTensor_(free)(tensor);

	THCudaBlas_shutdown(state);
	free(state);
	return 0;
}
