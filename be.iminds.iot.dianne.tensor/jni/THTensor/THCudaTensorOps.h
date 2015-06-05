#ifndef THTENSOR_CUDA_OPS_H
#define THTENSOR_CUDA_OPS_H

// define some additional operations for CUDATensors 

void THCudaTensor_dtanh(THCState *state, THCudaTensor *dst, THCudaTensor* src);
void THCudaTensor_sigoid(THCState *state, THCudaTensor *dst, THCudaTensor* src);
void THCudaTensor_dsigmoid(THCState *state, THCudaTensor *dst, THCudaTensor* src);
void THCudaTensor_threshold(THCState *state, THCudaTensor *dst, THCudaTensor* src, float thresh, float coeff, float of);
void THCudaTensor_dthreshold(THCState *state, THCudaTensor *dst, THCudaTensor* src, float thresh, float coeff);
void THCudaTensor_softmax(THCState *state, THCudaTensor *dst, THCudaTensor* src);

int THCudaTensor_argmax(THCState *state, THCudaTensor *t);
int THCudaTensor_argmin(THCState *state, THCudaTensor *t);

void THCudaTensor_spatialmaxpool(THCState *state, THCudaTensor *dst, THCudaTensor *src,
		int kW, int kH, int dW, int dH);
void THCudaTensor_spatialconvolve(THCState *state, THCudaTensor *dst, THCudaTensor *src,
		THCudaTensor* weight, THCudaTensor* bias, int dW, int dH);

#endif
