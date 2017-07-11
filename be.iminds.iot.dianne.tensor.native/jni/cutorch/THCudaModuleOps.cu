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

struct SELUupdateOutput_functor
{
  const float alpha_;
  const float lambda_;

  SELUupdateOutput_functor(float alpha, float lambda)
    : alpha_(alpha), lambda_(lambda)
  {}

  __device__ void operator()(float *output, const float *input) const
  {
    *output = *input <= 0 ? (exp(*input) - 1) * alpha_ * lambda_ : *input * lambda_;
  }
};


struct SELUupdateGradInput_functor
{
  const float alpha_;
  const float lambda_;

  SELUupdateGradInput_functor(float alpha, float lambda)
    : alpha_(alpha), lambda_(lambda)
  {}

  __device__ void operator()(float *gradInput, const float *output, const float *gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + alpha_ * lambda_)) : (*gradOutput * lambda_);
  }
};


void THCudaModule_selu(THCState *state, THCudaTensor *input, THCudaTensor *output,
  float alpha, float lambda)
{
  THAssertMsg(THCudaTensor_checkGPU(state, 2, input, output),
  "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.");

  THCudaTensor_resizeAs(state, output, input);
  THC_pointwiseApply2(state, output, input, SELUupdateOutput_functor(alpha, lambda));
}


void THCudaTensor_seluGradIn(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, THCudaTensor *output, float alpha, float lambda)
{
  THAssertMsg(THCudaTensor_checkGPU(state, 3, output, gradOutput, gradInput),
  "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.");

  THCudaTensor_resizeAs(state, gradInput, output);
  THC_pointwiseApply3(state, gradInput, output, gradOutput, SELUupdateGradInput_functor(alpha, lambda));
}
