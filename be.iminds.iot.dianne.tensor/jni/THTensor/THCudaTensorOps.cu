#ifndef THTENSOR_CUDA_OPS_H
#define THTENSOR_CUDA_OPS_H

// define some additional CUDA operations
extern "C" {
#include "THCudaTensorOps.h"
}
#include "THC/THCApply.cuh"
#include "THC/THCTensorMath.h"
#include "THC/THCBlas.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define SOFTMAX_THREADS 128
// Use 1024 threads per block, which requires cuda sm_2x or above
#ifdef JETSON
#define UNFOLD_THREADS 800
#else
#define UNFOLD_THREADS 1024
#endif

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + UNFOLD_THREADS - 1) / UNFOLD_THREADS;
}


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


__global__ void dmaxpool(float *input, float *output, float *goutput,
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
			float *ptr_output = output + yy*dH*input_w + xx*dW;
      		float *ptr_goutput = goutput + yy*output_w + xx;
      		
      		float max = -FLT_MAX;
      		int maxOffset = 0;
      		int kx, ky;
      		for(ky = 0; ky < kH; ky++) {
        		for(kx = 0; kx < kW; kx++) {
          			float val = ptr_input[kx];
          			if (val > max) {
            			max = val;
            			maxOffset = ky * input_w + kx;
          			}
        		}
        		ptr_input += input_w; // next input line
      		}
      		// Update output
      		*(ptr_output + maxOffset) = *ptr_goutput;
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

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
__global__ void im2col_kernel(const int n, const float* data_im,
    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col,
    float* data_col) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
      index < n;                                      
      index += blockDim.x * gridDim.x){
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_im[i * width + j] : 0;
          
        data_col += height_col * width_col;
      }
    }
  }
}


__global__ void scale2d(float *input, float *output,
                        int input_c, int input_h, int input_w,
                        int output_c, int output_h, int output_w,
                        float s_x, float s_y)
{
	int c = blockIdx.x;
	
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	int x = blockIdx.z*blockDim.x+threadIdx.x;
	
	if(y >= output_h){
		return;
	}
		
	if(x >= output_w){
		return;
	}
	
	float xx,yy,dx,dy;
	float v1,v2,v3,v4,v;
	int x1,x2,y1,y2,cc;
	
	yy = y*s_y;
	xx = x*s_x;

	// bilinear interpolation
	x1 = (int)xx;
	x2 = x1+1;
	if(x2==input_w)
		x2--;
	y1 = (int)yy;
	y2 = y1+1;
	if(y2==input_h)
		y2--;
	
	cc = c;	
	if(cc>=input_c)
		cc = 0;	

	v1 = input[cc*input_w*input_h + y1*input_w + x1];
	v2 = input[cc*input_w*input_h + y1*input_w + x2];
	v3 = input[cc*input_w*input_h + y2*input_w + x1];
	v4 = input[cc*input_w*input_h + y2*input_w + x2];

	dx = xx-x1;
	dy = yy-y1;

	v = v1*(1-dy)*(1-dx)
		+ v2 * (1-dy)*(dx)
	    + v3 * (dy)*(1-dx)
		+ v4 * (dx)*(dy);
	
	output[c*output_w*output_h + y*output_w + x] = v;
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
	
	void THCudaTensor_spatialdmaxpool(THCState *state, THCudaTensor *output, THCudaTensor *gradoutput,
		THCudaTensor *input, int kW, int kH, int dW, int dH){
	
		long nInputCols = input->size[2];
    	long nInputRows = input->size[1];
    	long nInputPlane = input->size[0];
    	long nOutputCols = (nInputCols - kW) / dW + 1;
    	long nOutputRows = (nInputRows - kH) / dH + 1;

    	input = THCudaTensor_newContiguous(state, input);
    	float* input_data = THCudaTensor_data(state, input);

    	THCudaTensor_resize3d(state, output, nInputPlane, nInputRows, nInputCols);
    	float* output_data = THCudaTensor_data(state, output);
    	
    	float* goutput_data = THCudaTensor_data(state, gradoutput);

    	// cuda blocks & threads:
    	int yblocks = (int)(16L / nInputPlane);
    	yblocks = yblocks < 1 ? 1 : yblocks;
    	dim3 blocks(nInputPlane,yblocks);
    	dim3 threads(32,8);

    	// run maxpool kernel
    	dmaxpool <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
    	  input_data, output_data, goutput_data,
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
	
	
	// helper function for unfolding matrix for convolution
	void im2col(THCState *state, const float* data_im, const int channels,
	    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
	    const int pad_w, const int stride_h, const int stride_w, float* data_col) {
	  // We are going to launch channels * height_col * width_col kernels, each
	  // kernel responsible for copying a single-channel grid.
	  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
	  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
	  int num_kernels = channels * height_col * width_col;
	  
	  // Launch
	  im2col_kernel <<<GET_BLOCKS(num_kernels), UNFOLD_THREADS, 0, THCState_getCurrentStream(state)>>> (
	      num_kernels, data_im, height, width, ksize_h, ksize_w,
	      pad_h, pad_w, stride_h, stride_w,
	      height_col, width_col, data_col
	  );
	}
	
	void THCudaTensor_spatialconvolve(THCState *state, THCudaTensor *output, THCudaTensor *input,
		THCudaTensor* weight, THCudaTensor* bias, int dW, int dH, int pW, int pH)
	{
		long nOutputPlane = weight->size[0];
		long kW = weight->size[3];
		long kH = weight->size[2];
		long inputWidth   = input->size[2];
  		long inputHeight  = input->size[1];
  		long nInputPlane = input->size[0];
  		long outputWidth  = (inputWidth + 2*pW - kW) / dW + 1;
  		long outputHeight = (inputHeight + 2*pH - kH) / dH + 1;
		
		// create temp tensors for unfolding
		THCudaTensor* columns = THCudaTensor_newWithSize2d(state, nInputPlane*kW*kH, outputHeight*outputWidth);
		THCudaTensor* ones = THCudaTensor_newWithSize2d(state, outputHeight, outputWidth);
		THCudaTensor_fill(state, ones, 1);
		
		
		// Do Bias first:
	    // M,N,K are dims of matrix A and B
	    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
	    long m_ = nOutputPlane;
	    long n_ = outputHeight * outputWidth;
	    long k_ = 1;
	
	    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
	    THCudaBlas_gemm(
	        state,
	        't', 'n',
	        n_, m_, k_,
	        1,
	        THCudaTensor_data(state, ones), k_,
	        THCudaTensor_data(state, bias), k_,
	        0,
	        THCudaTensor_data(state, output), n_
	    );
	
	    // Extract columns:
	    im2col(
	      state,
	      THCudaTensor_data(state, input),
	      nInputPlane, inputHeight, inputWidth, kH, kW, pH, pW, dH, dW,
	      THCudaTensor_data(state, columns)
	    );
	
	    // M,N,K are dims of matrix A and B
	    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
	    long m = weight->size[0];
	    long n = columns->size[1];
	    long k = columns->size[0];
	
	    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
	    THCudaBlas_gemm(
	        state,
	        'n', 'n',
	        n, m, k,
	        1,
	        THCudaTensor_data(state, columns), n,
	        THCudaTensor_data(state, weight), k,
	        1,
	        THCudaTensor_data(state, output), n
	    );
	  

		// free temp tensors
		THCudaTensor_free(state, columns);
		THCudaTensor_free(state, ones);
		
	}
	
	void THCudaTensor_spatialdinconvolve(THCState *state, THCudaTensor *gradInput, THCudaTensor *gradOutput,
		THCudaTensor* weight, int dW, int dH, int pW, int pH)
	{
		// TODO
	}
	
	void THCudaTensor_spatialdkerconvolve(THCState *state, THCudaTensor *gradKer, THCudaTensor *add,
		THCudaTensor* gradOutput, THCudaTensor* input, int dW, int dH, int pW, int pH)
	{
		// TODO
	}
	
	void THCudaTensor_scale2d(THCState *state, THCudaTensor *output, THCudaTensor *input)
	{
		long output_c, output_h, output_w;
		output_c = output->size[0];
		output_h = output->size[1];
		output_w = output->size[2];
		
		long input_c, input_h, input_w;
		if(input->nDimension==2){
			input_c = 1;
			input_h = input->size[0];
			input_w = input->size[1]; 
		} else {
			input_c = input->size[0];
			input_h = input->size[1];
			input_w = input->size[2];
		}
		
		input = THCudaTensor_newContiguous(state, input);
		
		float s_y = (input_h-1)/(float)(output_h-1);
		float s_x = (input_w-1)/(float)(output_h-1);
		
    	dim3 threads(16, 16);
    	dim3 blocks(output_c, output_h/threads.y + 1, output_w/threads.x + 1);
    	
    	scale2d <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
    	  THCudaTensor_data(state, input), THCudaTensor_data(state, output),
    	  input_c, input_h, input_w, output_c, output_h, output_w, s_x, s_y);
    	
    	  
    	THCudaTensor_free(state, input);
    	
	}
}
#endif
