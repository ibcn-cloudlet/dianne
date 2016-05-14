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
#include "TensorLoader.h"

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
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ker, jobject b, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, ker);
	THTensor* bias = getTensor(env, b);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);


	THNN_(SpatialConvolutionMM_updateOutput)(
	          state,
	          input,
	          output,
	          weight,
	          bias,
	          temp1,
	          temp2,
	          kW, kH,
	          dW, dH,
	          pW, pH);

	if(t1 == NULL){
		THTensor_(free)(
#ifdef CUDA
				state,
#endif
				temp1);
	}
	if(t2 == NULL){
		THTensor_(free)(
#ifdef CUDA
				state,
#endif
				temp2);
	}

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject ker, jobject in, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* weight = getTensor(env, ker);
	THTensor* input = getTensor(env, in);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

	THNN_(SpatialConvolutionMM_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          weight,
	          temp1,
	          temp2,
	          kW, kH,
	          dW, dH,
	          pW, pH);

	if(t1 == NULL){
		THTensor_(free)(
#ifdef CUDA
				state,
#endif
				temp1);
	}
	if(t2 == NULL){
		THTensor_(free)(
#ifdef CUDA
				state,
#endif
				temp2);
	}

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveAccGrad
  (JNIEnv * env, jclass c, jobject gradKer, jobject gradB, jobject gradOut, jobject in, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradWeight = getTensor(env, gradKer);
	THTensor* gradBias = getTensor(env, gradB);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

	THNN_(SpatialConvolutionMM_accGradParameters)(
	          state,
	          input,
	          gradOutput,
	          gradWeight,
	          gradBias,
	          temp1,
			  temp2,
			  kW, kH,
	          dW, dH,
	          pW, pH,
	          1.0);

	if(t1 == NULL){
		THTensor_(free)(
#ifdef CUDA
				state,
#endif
				temp1);
	}
	if(t2 == NULL){
		THTensor_(free)(
#ifdef CUDA
				state,
#endif
				temp2);
	}
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_batchnorm
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject w, jobject b, jobject rm, jobject rv, jobject sm, jobject sv, jboolean train){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor2d(env, out, input->size[0], input->size[1]);
	THTensor* weight = getTensor(env, w);
	THTensor* bias = getTensor(env, b);
	THTensor* running_mean = getTensor(env, rm);
	THTensor* running_var = getTensor(env, rv);
	THTensor* save_mean = getTensor(env, sm);
	THTensor* save_std = getTensor(env, sv);

	THNN_(BatchNormalization_updateOutput)(
	          state,
	          input,
	          output,
	          weight,
	          bias,
	          running_mean,
	          running_var,
	          save_mean,
	          save_std,
	          train,
	          0.1,
	          1e-5);

	return out == NULL ? createTensorObject(env, output) : out;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_batchnormGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject w, jobject rm, jobject rv, jobject sm, jobject sv, jboolean train){
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* gradInput = getTensor2d(env, gradIn, gradOutput->size[0], gradOutput->size[1]);
	THTensor* input = getTensor(env, in);
	THTensor* weight = getTensor(env, w);
	THTensor* running_mean = getTensor(env, rm);
	THTensor* running_var = getTensor(env, rv);
	THTensor* save_mean = getTensor(env, sm);
	THTensor* save_std = getTensor(env, sv);

	THNN_(BatchNormalization_backward)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          0,
	          0,
	          weight,
	          running_mean,
	          running_var,
	          save_mean,
	          save_std,
	          train,
	          0.1,
			  1e-5);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_batchnormAccGrad
  (JNIEnv * env, jclass c, jobject gradW, jobject gradB, jobject gradOut, jobject in, jobject rm, jobject rv, jobject sm, jobject sv, jboolean train){
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* gradWeight = getTensor(env, gradW);
	THTensor* gradBias = getTensor(env, gradB);
	THTensor* running_mean = getTensor(env, rm);
	THTensor* running_var = getTensor(env, rv);
	THTensor* save_mean = getTensor(env, sm);
	THTensor* save_std = getTensor(env, sv);

	THNN_(BatchNormalization_backward)(
	          state,
	          input,
	          gradOutput,
	          0,
	          gradWeight,
	          gradBias,
	          0,
	          running_mean,
	          running_var,
	          save_mean,
	          save_std,
	          train,
	          0.1,
			  1e-5);
}

