#ifndef THTENSOR_CUDA_OPS_H
#define THTENSOR_CUDA_OPS_H

// define some additional CUDA operations
extern "C" {
#include "THCudaTensorOps.h"
}
#include "THC/THCApply.cuh"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define SOFTMAX_THREADS 128

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

__global__ void maxpool(float *input, float *output,
                        int input_n, int input_h, int input_w,
                        int kH, int kW, int dH, int dW)
{
	// iterators
	int xx, yy;

	// output size
	const int output_w = (input_w - kW) / dW + 1;
	const int output_h = (input_h - kH) / dH + 1;

	// compute offsets based on thread/block ID
	int o = blockIdx.x;
	int i = o;

	int xx_start = threadIdx.x;
	int xx_end = output_w;
	const int xx_step = blockDim.x;

	int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
	int yy_end = output_h;
	const int yy_step = blockDim.y*gridDim.y;

	// select input/output plane
	output = output + o*output_w*output_h;
	input = input + i*input_w*input_h;

	// For all output pixels...
	for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    	for(xx = xx_start; xx < xx_end; xx+=xx_step) {
			float *ptr_input = input + yy*dH*input_w + xx*dW;
      		float *ptr_output = output + yy*output_w + xx;
      		float max = -FLT_MAX;
      		int kx, ky;
      		for(ky = 0; ky < kH; ky++) {
        		for(kx = 0; kx < kW; kx++) {
          			float val = ptr_input[kx];
          			if (val > max) {
            			max = val;
          			}
        		}
        		ptr_input += input_w; // next input line
      		}
      		// Update output
      		*ptr_output = max;
    	}
  	}
}


// softmax kernel with 128 threads based on cunn Softmax.cu
__global__ void softmax(float *output, float *input, int nframe, int dim)
{
  __shared__ float buffer[ SOFTMAX_THREADS + 1];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *output_k = output + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // max?
  buffer[threadIdx.x] = -FLT_MAX;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i];
    if(buffer[threadIdx.x] < z)
      buffer[threadIdx.x] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float max_k = -FLT_MAX;
    for (int i=0; i<blockDim.x; i++)
    {
      if(max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[SOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // sum?
  float max_k = buffer[SOFTMAX_THREADS];
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step) {
    float z = __expf(input_k[i]-max_k);
    buffer[threadIdx.x] += z;
    output_k[i] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[SOFTMAX_THREADS] = sum_k;
  }

  __syncthreads();

  // softmax
  float sum_k = buffer[SOFTMAX_THREADS];
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i] = output_k[i] / sum_k;
}


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
	
	int THCudaTensor_argmax(THCState *state, THCudaTensor *t){
		t = THCudaTensor_newContiguous(state, t);
		thrust::device_ptr<float> data(THCudaTensor_data(state, t));

		thrust::device_vector<float>::iterator iter =
			thrust::max_element(data, data + THCudaTensor_nElement(state, t));

		int position = thrust::device_pointer_cast(&(iter[0])) - data;
		THCudaTensor_free(state, t);

		return position;
	}

	int THCudaTensor_argmin(THCState *state, THCudaTensor *t){
		t = THCudaTensor_newContiguous(state, t);
		thrust::device_ptr<float> data(THCudaTensor_data(state, t));

		thrust::device_vector<float>::iterator iter =
			thrust::min_element(data, data + THCudaTensor_nElement(state, t));

		int position = thrust::device_pointer_cast(&(iter[0])) - data;
		THCudaTensor_free(state, t);

		return position;
	}
	
	void THCudaTensor_spatialmaxpool(THCState *state, THCudaTensor *output, THCudaTensor *input,
			int kW, int kH, int dW, int dH){	
		
		long nInputCols = input->size[2];
    	long nInputRows = input->size[1];
    	long nInputPlane = input->size[0];
    	long nOutputCols = (nInputCols - kW) / dW + 1;
    	long nOutputRows = (nInputRows - kH) / dH + 1;

    	input = THCudaTensor_newContiguous(state, input);
    	float* input_data = THCudaTensor_data(state, input);

    	THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);
    	float* output_data = THCudaTensor_data(state, output);

    	// cuda blocks & threads:
    	int yblocks = (int)(16L / nInputPlane);
    	yblocks = yblocks < 1 ? 1 : yblocks;
    	dim3 blocks(nInputPlane,yblocks);
    	dim3 threads(32,8);

    	// run maxpool kernel
    	maxpool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
    	  input_data, output_data,
    	  nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
	
		THCudaTensor_free(state, input);
	}
	
	
	void THCudaTensor_softmax(THCState *state, THCudaTensor *output, THCudaTensor *input){	
    	input = THCudaTensor_newContiguous(state, input);
    	float* input_data = THCudaTensor_data(state, input);
    	float* output_data = THCudaTensor_data(state, output);

    	// cuda blocks & threads:
    	dim3 blocks(1);
    	dim3 threads(SOFTMAX_THREADS);

    	// run softmax kernel
    	softmax<<<blocks,threads,0, THCState_getCurrentStream(state)>>>(
    		output_data, input_data, 1, input->storage->size);
    	
		THCudaTensor_free(state, input);
	}
}
#endif
