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
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
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
void THCudaTensor_spatialdmaxpool(THCState *state, THCudaTensor *dst, THCudaTensor *src2,
		THCudaTensor *src1, int kW, int kH, int dW, int dH);

void THCudaTensor_spatialconvolve(THCState *state, THCudaTensor *output, THCudaTensor *input,
		THCudaTensor* weight, THCudaTensor* bias, int dW, int dH, int pW, int pH);

void THCudaTensor_spatialdinconvolve(THCState *state, THCudaTensor *gradInput, THCudaTensor *gradOutput,
		THCudaTensor* weight, int dW, int dH, int pW, int pH);

void THCudaTensor_spatialdkerconvolve(THCState *state, THCudaTensor *gradKer, THCudaTensor *add,
		THCudaTensor* gradOutput, THCudaTensor* input, int dW, int dH, int pW, int pH);

void THCudaTensor_scale2d(THCState *state, THCudaTensor *dst, THCudaTensor *src);

#endif
