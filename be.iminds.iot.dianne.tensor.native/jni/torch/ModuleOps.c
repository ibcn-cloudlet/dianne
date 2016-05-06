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
#include "be_iminds_iot_dianne_tensor_ModuleOps.h"
#include "THNN.h"
#include "Tensor.h"
#include "TensorLoader.h"

// THNNState is void pointer for now?!
THNNState* state;

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_tanh
  (JNIEnv * env, jclass c, jobject out, jobject in){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);

	THNN_(Tanh_updateOutput)(
	          state,
	          input,
	          output);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_tanhDin
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject out){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* output = getTensor(env, out);

	THNN_(Tanh_updateGradInput)(
	          state,
	          0,
	          gradOutput,
	          gradInput,
	          output);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_sigmoid
  (JNIEnv *, jclass, jobject, jobject);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_sigmoidDin
  (JNIEnv *, jclass, jobject, jobject, jobject);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_threshold
  (JNIEnv *, jclass, jobject, jobject, jfloat, jfloat, jfloat);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_thresholdDin
  (JNIEnv *, jclass, jobject, jobject, jobject, jfloat, jfloat);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_softmax
  (JNIEnv *, jclass, jobject, jobject);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_softmaxDin
  (JNIEnv *, jclass, jobject, jobject, jobject);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialmaxpool
  (JNIEnv *, jclass, jobject, jobject, jint, jint, jint, jint, jint, jint);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialmaxpoolDin
  (JNIEnv *, jclass, jobject, jobject, jobject, jint, jint, jint, jint, jint, jint);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolve
  (JNIEnv *, jclass, jobject, jobject, jobject, jobject, jint, jint, jint, jint);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveDin
  (JNIEnv *, jclass, jobject, jobject, jobject, jint, jint, jint, jint);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveDker
  (JNIEnv *, jclass, jobject, jobject, jobject, jobject, jint, jint, jint, jint);

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveDbias
  (JNIEnv *, jclass, jobject, jobject);
