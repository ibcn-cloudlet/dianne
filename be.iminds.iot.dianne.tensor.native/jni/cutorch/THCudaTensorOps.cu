/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
// added to be able to compile with default c++ compiler and cross compilers
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
// define some additional CUDA operations
#include "THCudaTensorOps.h"
#include "THCApply.cuh"
#include "THCTensorMath.h"
#include "THCBlas.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

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


__global__ void rotate(float *input, float *output,
                        int channels, int height, int width,
                        double sin_theta, double cos_theta, float center_x,
                        float center_y, int zeropad)
{
	int c = blockIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	int x = blockIdx.z*blockDim.x+threadIdx.x;

	int heightIndex = (int)((x - center_x)*sin_theta + (y - center_y)*cos_theta + center_y);
	int widthIndex = (int)((x - center_x)*cos_theta - (y - center_y)*sin_theta + center_x);

	float v;
	
	if(zeropad != 0) {
		if(heightIndex < 0 || widthIndex < 0
			|| heightIndex >= height || widthIndex >= width){
			v = 0.0f;
		} else {
			v = input[c*width*height + heightIndex*width + widthIndex];
		}
	} else {
		// use boundary values to extend?
		if(heightIndex < 0) {
			heightIndex = 0;
		} else if(heightIndex >= height) {
			heightIndex = height - 1;
		}

		if(widthIndex < 0 ) {
			widthIndex = 0;
		} else if(widthIndex >= width) {
			widthIndex = width - 1;
		}

		v = input[c*width*height + heightIndex*width + widthIndex];
	}
	
	output[c*width*height + y*width + x] = v;
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


void THCudaTensor_rotate(THCState *state, THCudaTensor *output, THCudaTensor *input,
		float theta, float center_x, float center_y, int zeropad )
{
	long channels, height, width;
	channels = input->size[0];
	height = input->size[1];
	width = input->size[2];
	input = THCudaTensor_newContiguous(state, input);
	
	double sin_theta = sin(theta);
	double cos_theta = cos(theta);
	
	dim3 threads(16, 16);
	dim3 blocks(channels, height/threads.y + 1, width/threads.x + 1);
	
	rotate <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
	  THCudaTensor_data(state, input), THCudaTensor_data(state, output),
	  channels, height, width, sin_theta, cos_theta, center_x, center_y, zeropad);
	
	THCudaTensor_free(state, input);
}
