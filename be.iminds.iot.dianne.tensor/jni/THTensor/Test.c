
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

#ifdef CUDA
	THCState* state = (THCState*)malloc(sizeof(THCState));
	THCudaInit(state);

	printf("Initialized CUDA %d %d \n", state->numDevices, state->numUserStreams);
#endif


	THTensor* tensor = THTensor_(newWithSize2d)(
#ifdef CUDA
		state,
#endif
		2, 2);


	THTensor_(fill)(
#ifdef CUDA
		state,
#endif
		tensor, 1.0f);

#ifdef CUDA
	long size = tensor->storage->size*sizeof(real);
	real* data = (real*)malloc(size);
	real* dataGPU = THTensor_(data)(
		state,
		tensor);
	cudaMemcpy(data, dataGPU, size, cudaMemcpyDeviceToHost);
#else 
	real* data = THTensor_(data)(tensor);
#endif
	printf("Tensor: \n");
	printf(" %f %f \n %f %f \n", data[0], data[1], data[2], data[3]);
#ifdef CUDA
	free(data);
#endif

	THTensor_(add)(
#ifdef CUDA
		state,
#endif
		tensor, tensor, 1.0f);

#ifdef CUDA
	data = (real*)malloc(size);
 	dataGPU = THTensor_(data)(state, tensor);
	cudaMemcpy(data, dataGPU, size, cudaMemcpyDeviceToHost);
#else
	data = THTensor_(data)(tensor);
#endif
	printf("Add 1: \n");
	printf(" %f %f \n %f %f \n", data[0], data[1], data[2], data[3]);
#ifdef CUDA
	free(data);
#endif


	THTensor_(free)(
#ifdef CUDA
		state,
#endif
		tensor);

	printf("Test dot\n");

	THTensor* v1 = THTensor_(newWithSize1d)(
#ifdef CUDA
		state,
#endif
		4);

	THTensor_(fill)(
#ifdef CUDA
		state,
#endif
		v1, 1.0f);

	THTensor* v2 = THTensor_(newWithSize1d)(
#ifdef CUDA
		state,
#endif
		4);

	THTensor_(fill)(
#ifdef CUDA
		state,
#endif
		v2, 1.0f);


	real dot = THTensor_(dot)(
#ifdef CUDA
			state,
#endif
			v1, v2);

	printf("Dot (expect 4): %f \n", dot);

	THTensor_(free)(
#ifdef CUDA
		state,
#endif
		v1);

	THTensor_(free)(
#ifdef CUDA
		state,
#endif
		v2);
#ifdef CUDA
	THCudaBlas_shutdown(state);
	free(state);
#endif
	return 0;
}
