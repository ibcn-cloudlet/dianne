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
#ifndef THTENSOR_CUDA_OPS_H
#define THTENSOR_CUDA_OPS_H

#include "THC.h"
// define some additional operations for CUDATensors

int THCudaTensor_argmax(THCState *state, THCudaTensor *t);
int THCudaTensor_argmin(THCState *state, THCudaTensor *t);
void THCudaTensor_scale2d(THCState *state, THCudaTensor *dst, THCudaTensor *src);
void THCudaTensor_rotate(THCState *state, THCudaTensor *dst, THCudaTensor *src, float theta, float center_x, float center_y, int zeropad);


#endif
