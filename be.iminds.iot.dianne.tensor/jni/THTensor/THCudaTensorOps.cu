#ifndef THTENSOR_CUDA_OPS_H
#define THTENSOR_CUDA_OPS_H

// define some additional CUDA operations
extern "C" {
#include "THCudaTensorOps.h"
}
#include "THCudaTensorJNI.h"
#include "THC/THCApply.cuh"

struct TensorDTanOp {
	  __device__ __forceinline__ void operator()(float* out, float* in) {
	    *out = 1.- (*in) * (*in);
	  }

	  __device__ __forceinline__ void operator()(float* v) {
	    *v = 1.- (*v) * (*v);
	  }
};


extern "C" {
	void THCudaTensor_dtanh(THCState *state, THCudaTensor *dest, THCudaTensor *src)
	{
		THAssert(THCudaTensor_checkGPU(state, 2, dest, src));
		if (dest == src) {
			if (!THCudaTensor_pointwiseApply1(state, dest, TensorDTanOp())) {
				THArgCheck(false, 2, CUTORCH_DIM_WARNING);
			}
		} else {
			THCudaTensor_resizeAs(state, dest, src);
	
			if (!THCudaTensor_pointwiseApply2(state, dest, src, TensorDTanOp())) {
				THArgCheck(false, 2, CUTORCH_DIM_WARNING);
			}
		}
	
		THCudaCheck(cudaGetLastError());
	}
}

#endif
