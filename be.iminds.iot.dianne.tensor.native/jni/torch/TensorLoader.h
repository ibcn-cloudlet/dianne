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
#ifndef TENSOR_LOADER_H
#define TENSOR_LOADER_H

#ifdef CUDA
#include "CudaTensor.h"
#else
#include "Tensor.h"
#endif

jfieldID TENSOR_ADDRESS_FIELD;
jmethodID TENSOR_INIT;
jclass TENSOR_CLASS;

// convert object to a Tensor - create new one if NULL
THTensor* getTensor(JNIEnv* env, jobject o);
// convert object to a Tensor - resize to given dims
THTensor* getTensor1d(JNIEnv* env, jobject o, int d0);
THTensor* getTensor2d(JNIEnv* env, jobject o, int d0, int d1);
THTensor* getTensor3d(JNIEnv* env, jobject o, int d0, int d1, int d2);
THTensor* getTensor4d(JNIEnv* env, jobject o, int d0, int d1, int d2, int d3);

jobject createTensorObject(JNIEnv* env, THTensor* t);

#endif
