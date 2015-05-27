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

struct TensorThresholdOp {
	  TensorThresholdOp(float t, float c, float o) : thresh(t),coeff(c),offset(o) {}

	  __device__ __forceinline__ void operator()(float* out, float* in) {
	    *out = (*in) > thresh ? (*in) : coeff * (*in) + offset;
	  }

	  __device__ __forceinline__ void operator()(float* v) {
	    *v = (*v) > thresh ? (*v) : coeff * (*v) + offset;
	  }
	  
	  const float thresh;
	  const float coeff;
	  const float offset;
};

struct TensorDThresholdOp {
	  TensorDThresholdOp(float t, float c) : thresh(t),coeff(c) {}

	  __device__ __forceinline__ void operator()(float* out, float* in) {
	    *out = (*in) > thresh ? 1 : coeff;
	  }

	  __device__ __forceinline__ void operator()(float* v) {
	    *v = (*v) > thresh ? 1 : coeff;
	  }
	  
	  const float thresh;
	  const float coeff;
};

struct TensorExpMinusOp {
	  TensorExpMinusOp(float m) : minus(m) {}

	  __device__ __forceinline__ void operator()(float* out, float* in) {
	    *out = exp( (*in) - minus);
	  }

	  __device__ __forceinline__ void operator()(float* v) {
	    *v = exp( (*v) - minus);
	  }
	  
	  const float minus;
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
	
	void THCudaTensor_threshold(THCState *state, THCudaTensor *dest, THCudaTensor* src, float thresh, float coeff, float of){
		if (dest == src) {
			THCudaTensor_pointwiseApply1(state, dest, TensorThresholdOp(thresh, coeff, of));
		} else {
			THCudaTensor_pointwiseApply2(state, dest, src, TensorThresholdOp(thresh, coeff, of));
		}
	}

	void THCudaTensor_dthreshold(THCState *state, THCudaTensor *dest, THCudaTensor* src, float thresh, float coeff){
		if (dest == src) {
			THCudaTensor_pointwiseApply1(state, dest, TensorDThresholdOp(thresh, coeff));
		} else {
			THCudaTensor_pointwiseApply2(state, dest, src, TensorDThresholdOp(thresh, coeff));
		}
	}
	
	void THCudaTensor_expminus(THCState *state, THCudaTensor *dest, THCudaTensor* src, float min){
		if (dest == src) {
			THCudaTensor_pointwiseApply1(state, dest, TensorExpMinusOp(min));
		} else {
			THCudaTensor_pointwiseApply2(state, dest, src, TensorExpMinusOp(min));
		}
	}
	
	
}
#endif
