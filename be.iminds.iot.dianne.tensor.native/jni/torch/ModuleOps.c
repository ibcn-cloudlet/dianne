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

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_tanhGradIn
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
  (JNIEnv * env, jclass c, jobject out, jobject in){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);

	THNN_(Sigmoid_updateOutput)(
	          state,
	          input,
	          output);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_sigmoidGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject out){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* output = getTensor(env, out);

	THNN_(Sigmoid_updateGradInput)(
	          state,
	          0,
	          gradOutput,
	          gradInput,
	          output);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_threshold
  (JNIEnv * env, jclass c, jobject out, jobject in, jfloat threshold, jfloat val){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);

	THNN_(Threshold_updateOutput)(
	          state,
	          input,
	          output,
			  threshold,
			  val,
			  0);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_thresholdGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jfloat threshold){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);

	THNN_(Threshold_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          threshold,
			  0);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_prelu
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject w, jint nOutputPlane){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, w);

	THNN_(PReLU_updateOutput)(
			  state,
	          input,
	          output,
	          weight,
	          nOutputPlane);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_preluGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject w, jint nOutputPlane){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* weight = getTensor(env, w);

	THNN_(PReLU_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          weight,
	          nOutputPlane);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_preluAccGrad
  (JNIEnv * env, jclass c, jobject gradW, jobject gradOut, jobject in, jobject w, jint nOutputPlane){
	THTensor* gradWeight = getTensor(env, gradW);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* weight = getTensor(env, w);

	THNN_(PReLU_accGradParameters)(
	          state,
	          input,
	          gradOutput,
	          0, // not used?!
	          weight,
	          gradWeight,
	          0, // not used?!
	          0, // not used?!
	          nOutputPlane,
	          1);
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_softmax
  (JNIEnv * env, jclass c, jobject out, jobject in){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);

	THNN_(SoftMax_updateOutput)(
	          state,
	          input,
	          output);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_softmaxGradIn
(JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject out){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* output = getTensor(env, out);

	THNN_(SoftMax_updateGradInput)(
	          state,
	          0,
	          gradOutput,
	          gradInput,
	          output);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialmaxpool
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ind, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* indices = getTensor(env, ind);

	THNN_(SpatialMaxPooling_updateOutput)(
	          state,
	          input,
	          output,
	          indices,
	          kW, kH,
	          dW, dH,
	          pW, pH,
	          0);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialmaxpoolGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject ind, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* indices = getTensor(env, ind);

	THNN_(SpatialMaxPooling_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          indices,
	          kW, kH,
	          dW, dH,
	          pW, pH,
	          0);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolve
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ker, jobject b, jobject fin, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, ker);
	THTensor* bias = getTensor(env, b);
	THTensor* finput = getTensor(env, fin);

	THNN_(SpatialConvolutionMM_updateOutput)(
	          state,
	          input,
	          output,
	          weight,
	          bias,
	          finput,
	          0, // fgradInput not used here?!
	          kW, kH,
	          dW, dH,
	          pW, pH);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject ker, jobject in, jobject fin, jobject fgradIn, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* weight = getTensor(env, ker);
	THTensor* input = getTensor(env, in);
	THTensor* finput = getTensor(env, fin);
	THTensor* fgradInput = getTensor(env, fgradIn);

	THNN_(SpatialConvolutionMM_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          weight,
	          finput,
	          fgradInput,
	          kW, kH,
	          dW, dH,
	          pW, pH);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveAccGrad
  (JNIEnv * env, jclass c, jobject gradKer, jobject gradB, jobject gradOut, jobject in, jobject fin, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradWeight = getTensor(env, gradKer);
	THTensor* gradBias = getTensor(env, gradB);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* finput = getTensor(env, fin);

	THNN_(SpatialConvolutionMM_accGradParameters)(
	          state,
	          input,
	          gradOutput,
	          gradWeight,
	          gradBias,
	          finput,
	          0,  // fgradInput not used here?!
	          kW, kH,
	          dW, dH,
	          pW, pH,
	          1);
}
