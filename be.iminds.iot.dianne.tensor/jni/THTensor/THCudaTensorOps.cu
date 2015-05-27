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

struct TensorSigmoidOp {
	  __device__ __forceinline__ void operator()(float* out, float* in) {
	    *out = 1./(1.+ exp(- *in));
	  }

	  __device__ __forceinline__ void operator()(float* v) {
	    *v = 1./(1.+ exp(- *v));
	  }
};

struct TensorDSigmoidOp {
	  __device__ __forceinline__ void operator()(float* out, float* in) {
	    *out = (1. - *in) * (*in);
	  }

	  __device__ __forceinline__ void operator()(float* v) {
	    *v = (1. - *v) * (*v);
	  }
};


extern "C" {
	void THCudaTensor_dtanh(THCState *state, THCudaTensor *dest, THCudaTensor *src)
	{
		if (dest == src) {
			THCudaTensor_pointwiseApply1(state, dest, TensorDTanOp());
		} else {
			THCudaTensor_pointwiseApply2(state, dest, src, TensorDTanOp());
		}
	}
	
	void THCudaTensor_sigmoid(THCState *state, THCudaTensor *dest, THCudaTensor *src)
	{
		if (dest == src) {
			THCudaTensor_pointwiseApply1(state, dest, TensorSigmoidOp());
		} else {
			THCudaTensor_pointwiseApply2(state, dest, src, TensorSigmoidOp());
		}
	}
	
	void THCudaTensor_dsigmoid(THCState *state, THCudaTensor *dest, THCudaTensor *src)
	{
		if (dest == src) {
			THCudaTensor_pointwiseApply1(state, dest, TensorDSigmoidOp());
		} else {
			THCudaTensor_pointwiseApply2(state, dest, src, TensorDSigmoidOp());
		}
	}
}

#endif
