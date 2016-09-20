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

#ifdef CUDA
THCState* state;
#else
THNNState* state;
#endif

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



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_logsoftmax
  (JNIEnv * env, jclass c, jobject out, jobject in){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);

	THNN_(LogSoftMax_updateOutput)(
	          state,
	          input,
	          output);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_logsoftmaxGradIn
(JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject out){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* output = getTensor(env, out);

	THNN_(LogSoftMax_updateGradInput)(
	          state,
	          0,
	          gradOutput,
	          gradInput,
	          output);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_temporalmaxpool
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ind, jint kW, jint dW){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* indices = getTensor(env, ind);

	THNN_(TemporalMaxPooling_updateOutput)(
		          state,
		          input,
		          output,
		          indices,
		          kW,
		          dW);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_temporalmaxpoolGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject ind, jint kW, jint dW){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* indices = getTensor(env, ind);

	THNN_(TemporalMaxPooling_updateGradInput)(
		          state,
		          input,
		          gradOutput,
		          gradInput,
		          indices,
		          kW,
		          dW);

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


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricmaxpool
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ind, jint kW, jint kH, jint kD,
		  jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* indices = getTensor(env, ind);

	THNN_(VolumetricMaxPooling_updateOutput)(
	          state,
	          input,
	          output,
	          indices,
	          kD, kW, kH,
	          dD, dW, dH,
	          pD, pW, pH,
	          0);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricmaxpoolGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject ind,
		  jint kW, jint kH, jint kD, jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* indices = getTensor(env, ind);

	THNN_(VolumetricMaxPooling_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          indices,
	          dD, dW, dH,
	          pD, pW, pH);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialmaxunpool
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ind, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* indices = getTensor(env, ind);

	int dimh = 1;
	int dimw = 2;

	if (input->nDimension == 4)
	{
	  dimh++;
	  dimw++;
	}

	int iH = input->size[dimh];
	int iW = input->size[dimw];

	int oH = (iH - 1) * dH + kH - 2* pH;
	int oW = (iW - 1) * dW + kW - 2* pW;

	THNN_(SpatialMaxUnpooling_updateOutput)(
	          state,
	          input,
	          output,
	          indices,
	          oW, oH);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialmaxunpoolGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject ind, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* indices = getTensor(env, ind);

	int dimh = 1;
	int dimw = 2;

	if (input->nDimension == 4)
	{
	  dimh++;
	  dimw++;
	}

	int iH = input->size[dimh];
	int iW = input->size[dimw];

	int oH = (iH - 1) * dH + kH - 2* pH;
	int oW = (iW - 1) * dW + kW - 2* pW;

	THNN_(SpatialMaxUnpooling_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          indices,
	          oW, oH);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricmaxunpool
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ind, jint kW, jint kH, jint kD,
		  jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* indices = getTensor(env, ind);

	int dimd = 1;
	int dimh = 2;
	int dimw = 3;

	if (input->nDimension == 5)
	{
	  dimd++;
	  dimh++;
	  dimw++;
	}

	int iD = input->size[dimd];
	int iH = input->size[dimh];
	int iW = input->size[dimw];

	int oD = (iD - 1) * dD + kD - 2* pD;
	int oH = (iH - 1) * dH + kH - 2* pH;
	int oW = (iW - 1) * dW + kW - 2* pW;

	THNN_(VolumetricMaxUnpooling_updateOutput)(
	          state,
	          input,
	          output,
	          indices,
	          oD, oW, oH,
	          dD, dW, dH,
	          pD, pW, pH);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricmaxunpoolGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject ind,
		  jint kW, jint kH, jint kD, jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* indices = getTensor(env, ind);

	int dimd = 1;
	int dimh = 2;
	int dimw = 3;

	if (input->nDimension == 5)
	{
	  dimd++;
	  dimh++;
	  dimw++;
	}

	int iD = input->size[dimd];
	int iH = input->size[dimh];
	int iW = input->size[dimw];

	int oD = (iD - 1) * dD + kD - 2* pD;
	int oH = (iH - 1) * dH + kH - 2* pH;
	int oW = (iW - 1) * dW + kW - 2* pW;

	THNN_(VolumetricMaxUnpooling_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          indices,
			  oD, oW, oH,
	          dD, dW, dH,
	          pD, pW, pH);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialavgpool
  (JNIEnv * env, jclass c, jobject out, jobject in, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH, jboolean ceil, jboolean count_pad){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);

	THNN_(SpatialAveragePooling_updateOutput)(
	          state,
	          input,
	          output,
	          kW, kH,
	          dW, dH,
	          pW, pH,
	          ceil,
			  count_pad);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialavgpoolGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH, jboolean ceil, jboolean count_pad){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);

	THNN_(SpatialAveragePooling_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          kW, kH,
	          dW, dH,
	          pW, pH,
	          ceil,
			  count_pad);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricavgpool
  (JNIEnv * env, jclass c, jobject out, jobject in, jint kW, jint kH, jint kD, jint dW, jint dH, jint dD){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);

	THNN_(VolumetricAveragePooling_updateOutput)(
	          state,
	          input,
	          output,
	          kD, kW, kH,
	          dD, dW, dH);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricavgpoolGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jint kW, jint kH, jint kD, jint dW, jint dH, jint dD){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);

	THNN_(VolumetricAveragePooling_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          kD, kW, kH,
	          dD, dW, dH);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_temporalconvolve
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ker, jobject b, jint kW, jint dW, jint nI, jint nO){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, ker);
	THTensor* bias = getTensor(env, b);

	THNN_(TemporalConvolution_updateOutput)(
	          state,
	          input,
	          output,
	          weight,
	          bias,
	          kW, dW,
			  nI, nO);

	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_temporalconvolveGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject ker, jobject in, jint kW, jint dW){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* weight = getTensor(env, ker);
	THTensor* input = getTensor(env, in);

	THNN_(TemporalConvolution_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          weight,
	          kW, dW);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_temporalconvolveAccGrad
  (JNIEnv * env, jclass c, jobject gradKer, jobject gradB, jobject gradOut, jobject in, jint kW, jint dW){
	THTensor* gradWeight = getTensor(env, gradKer);
	THTensor* gradBias = getTensor(env, gradB);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);

	THNN_(TemporalConvolution_accGradParameters)(
	          state,
	          input,
	          gradOutput,
	          gradWeight,
	          gradBias,
			  kW, dW,
	          1.0);
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


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricconvolve
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ker, jobject b, jobject t1, jobject t2,
		  jint kW, jint kH, jint kD, jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, ker);
	THTensor* bias = getTensor(env, b);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);


#ifdef CUDA
	// VolumericConvolution expects other dimensions than MM
	if(weight->nDimension == 2){
		int nOutputPlanes = weight->size[0];
		int nInputPlanes = weight->size[1]/(kD*kH*kW);
		THTensor_(resize5d)(
#ifdef CUDA
			state,
#endif
			weight, nOutputPlanes, nInputPlanes, kD, kH, kW);
	}


	THNN_(VolumetricConvolution_updateOutput)(
	          state,
	          input,
	          output,
	          weight,
	          bias,
	          temp1,
	          temp2,
	          dD, dW, dH,
			  pD, pW, pH);
#else
	// use ConvolutionMM on CPU
	THNN_(VolumetricConvolutionMM_updateOutput)(
			state,
	        input,
	        output,
	        weight,
	        bias,
	        temp1,
	        kD, kW, kH,
	        dD, dW, dH,
	        pD, pW, pH);
#endif

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

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricconvolveGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject ker, jobject in, jobject t1, jobject t2,
		  jint kW, jint kH, jint kD, jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* weight = getTensor(env, ker);
	THTensor* input = getTensor(env, in);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

#ifdef CUDA
	// VolumericConvolution expects other dimensions than MM
	if(weight->nDimension == 2){
		int nOutputPlanes = weight->size[0];
		int nInputPlanes = weight->size[1]/(kD*kH*kW);
		THTensor_(resize5d)(
#ifdef CUDA
			state,
#endif
			weight, nOutputPlanes, nInputPlanes, kD, kH, kW);
	}

	THNN_(VolumetricConvolution_updateGradInput)(
	          state,
	          input,
	          gradOutput,
	          gradInput,
	          weight,
	          temp1,
	          dD, dW, dH,
	          pD, pW, pH);
#else
	THNN_(VolumetricConvolutionMM_updateGradInput)(
		state,
	    input,
	    gradOutput,
	    gradInput,
	    weight,
	    temp1,
	    temp2,
		kD, kW, kH,
		dD, dW, dH,
		pD, pW, pH);
#endif

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

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricconvolveAccGrad
  (JNIEnv * env, jclass c, jobject gradKer, jobject gradB, jobject gradOut, jobject in, jobject t1, jobject t2,
		  jint kW, jint kH, jint kD, jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* gradWeight = getTensor(env, gradKer);
	THTensor* gradBias = getTensor(env, gradB);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

#ifdef CUDA
	// VolumericConvolution expects other dimensions than MM
	if(gradWeight->nDimension == 2){
		int nOutputPlanes = gradWeight->size[0];
		int nInputPlanes = gradWeight->size[1]/(kD*kH*kW);
		THTensor_(resize5d)(
#ifdef CUDA
			state,
#endif
			gradWeight, nOutputPlanes, nInputPlanes, kD, kH, kW);
	}

	THNN_(VolumetricConvolution_accGradParameters)(
	          state,
	          input,
	          gradOutput,
	          gradWeight,
	          gradBias,
	          temp1,
			  temp2,
	          dD, dW, dH,
	          pD, pW, pH,
	          1.0);
#else
	THNN_(VolumetricConvolutionMM_accGradParameters)(
		state,
	    input,
	    gradOutput,
	    gradWeight,
	    gradBias,
	    temp1,
	    1.0);
#endif


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



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialfullconvolve
(JNIEnv * env, jclass c, jobject out, jobject in, jobject ker, jobject b, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, ker);
	THTensor* bias = getTensor(env, b);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

	THNN_(SpatialFullConvolution_updateOutput)(
		state,
	    input,
	    output,
	    weight,
	    bias,
	    temp1,
	    temp2,
	    kW, kH,
	    dW, dH,
	    pW, pH,
	    0, 0);

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

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialfullconvolveGradIn
(JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject ker, jobject in, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* weight = getTensor(env, ker);
	THTensor* input = getTensor(env, in);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

	THNN_(SpatialFullConvolution_updateGradInput)(
		state,
	    input,
	    gradOutput,
	    gradInput,
	    weight,
	    temp1,
	    kW, kH,
	    dW, dH,
	    pW, pH,
	    0, 0);

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

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialfullconvolveAccGrad
(JNIEnv * env, jclass c, jobject gradKer, jobject gradB, jobject gradOut, jobject in, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradWeight = getTensor(env, gradKer);
	THTensor* gradBias = getTensor(env, gradB);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

	THNN_(SpatialFullConvolution_accGradParameters)(
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
	        0, 0,
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



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricfullconvolve
(JNIEnv * env, jclass c, jobject out, jobject in, jobject ker, jobject b, jobject t1, jobject t2,
		  jint kW, jint kH, jint kD, jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, ker);
	THTensor* bias = getTensor(env, b);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

	THNN_(VolumetricFullConvolution_updateOutput)(
	 	state,
	    input,
	    output,
	    weight,
	    bias,
	    temp1,
		temp2,
		dD, dW, dH,
	    pD, pW, pH,
	    0, 0, 0);


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

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricfullconvolveGradIn
(JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject ker, jobject in, jobject t1, jobject t2,
		  jint kW, jint kH, jint kD, jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* weight = getTensor(env, ker);
	THTensor* input = getTensor(env, in);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

	THNN_(VolumetricFullConvolution_updateGradInput)(
		state,
	    input,
	    gradOutput,
	    gradInput,
	    weight,
		temp1,
	    temp2,
	    dD, dW, dH,
		pD, pW, pH,
	    0, 0, 0);


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

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_volumetricfullconvolveAccGrad
(JNIEnv * env, jclass c, jobject gradKer, jobject gradB, jobject gradOut, jobject in, jobject t1, jobject t2,
		  jint kW, jint kH, jint kD, jint dW, jint dH, jint dD, jint pW, jint pH, jint pD){
	THTensor* gradWeight = getTensor(env, gradKer);
	THTensor* gradBias = getTensor(env, gradB);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* temp1 = getTensor(env, t1);
	THTensor* temp2 = getTensor(env, t2);

	THNN_(VolumetricFullConvolution_accGradParameters)(
			state,
			input,
			gradOutput,
			gradWeight,
			gradBias,
			temp1,
			temp2,
			dD, dW, dH,
			pD, pW, pH,
			0, 0, 0,
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
	THTensor* output;
	if(input->nDimension == 2)
		output = getTensor2d(env, out, input->size[0], input->size[1]);
	else
		output = getTensor3d(env, out, input->size[0], input->size[1], input->size[2]);

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

	THTensor* gradInput;
	if(gradOutput->nDimension == 2)
		gradInput = getTensor2d(env, gradIn, gradOutput->size[0], gradOutput->size[1]);
	else
		gradInput = getTensor3d(env, gradIn, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);

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
	          1.0,
		  1e-5);

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_batchnormAccGrad
  (JNIEnv * env, jclass c, jobject gradW, jobject gradB, jobject gradOut, jobject in, jobject w, jobject rm, jobject rv, jobject sm, jobject sv, jboolean train){
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* weight = getTensor(env, w);
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
	          weight,
	          running_mean,
	          running_var,
	          save_mean,
	          save_std,
	          train,
	          1.0,
		  1e-5);
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_linear
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject w, jobject b, jobject o){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, w);
	THTensor* bias = getTensor(env, b);
	THTensor* ones = getTensor(env, o);

	if(input->nDimension % 2 == 1){
		// 1d or 3d tensor, treat as one input by default
		if(input->nDimension == 3){
			THTensor_(resize1d)(
#ifdef CUDA
				state,
#endif
				input, input->size[0]*input->size[1]*input->size[2]);
		}

		THTensor_(addmv)(
#ifdef CUDA
				state,
#endif
				output, 1.0f, bias, 1.0f, weight, input);
	} else {
		// 2d or 4d tensor, treat as batch by default
		if(input->nDimension == 4){
			THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			input, input->size[0],input->size[1]*input->size[2]*input->size[3]);
		}

		THTensor_(resize2d)(
#ifdef CUDA
			state,
#endif
			output,
			input->size[0], weight->size[0]);


		THTensor_(transpose)(
#ifdef CUDA
			state,
#endif
			weight, NULL, 0, 1);

		THTensor_(addmm)(
#ifdef CUDA
			state,
#endif
			output, 0.0f, output, 1.0f, input, weight);

		THTensor_(addr)(
#ifdef CUDA
			state,
#endif
			output, 1.0f, output, 1.0f, ones, bias);

		THTensor_(transpose)(
#ifdef CUDA
			state,
#endif
			weight, NULL, 0, 1);

	}
	return out == NULL ? createTensorObject(env, output) : out;
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_linearGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject w, jobject in){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* weight = getTensor(env, w);
	THTensor* input = getTensor(env, in);

	THTensor_(resizeAs)(
#ifdef CUDA
		state,
#endif
		gradInput, input);

	if(input->nDimension % 2 == 1){
		// treat as vector input
		THTensor_(transpose)(
#ifdef CUDA
			state,
#endif
			weight, NULL, 0, 1);

		THTensor_(addmv)(
#ifdef CUDA
			state,
#endif
			gradInput, 0.0f, gradInput, 1.0f, weight, gradOutput);

		THTensor_(transpose)(
#ifdef CUDA
			state,
#endif
			weight, NULL, 0, 1);
	} else{
		// treat as batch input

		THTensor_(addmm)(
#ifdef CUDA
			state,
#endif
			gradInput, 0.0f, gradInput, 1.0f, gradOutput, weight);
	}

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}

JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_linearAccGrad
  (JNIEnv * env, jclass c, jobject gradW, jobject gradB, jobject gradOut, jobject in, jobject o){
	THTensor* gradWeight = getTensor(env, gradW);
	THTensor* gradBias = getTensor(env, gradB);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* ones = getTensor(env, o);

	if(input->nDimension % 2 == 1){
		THTensor_(addr)(
#ifdef CUDA
			state,
#endif
			gradWeight, 1.0f, gradWeight, 1.0f, gradOutput, input);

		THTensor_(cadd)(
#ifdef CUDA
			state,
#endif
			gradBias, gradBias, 1, gradOutput);
	} else {
		// batched input
		// gradWeight
		THTensor_(transpose)(
#ifdef CUDA
			state,
#endif
			gradOutput, NULL, 0, 1);

		THTensor_(addmm)(
#ifdef CUDA
			state,
#endif
			gradWeight, 1.0f, gradWeight, 1.0f, gradOutput, input);

		// gradBias
		THTensor_(addmv)(
#ifdef CUDA
			state,
#endif
			gradBias, 1.0f, gradBias, 1.0f, gradOutput, ones);

		THTensor_(transpose)(
#ifdef CUDA
			state,
#endif
			gradOutput, NULL, 0, 1);

	}

}
