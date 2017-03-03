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

#include "CudnnTensor.h"

#include <time.h>
clock_t start, end;
double cpu_time_used;


cudnnHandle_t cudnnHandle;

// optionally fix convolution algorithms
int convFwAlg = -1;
int convBwAlg = -1;
int convAgAlg = -1;


// to be used when we use a shared workspace (by default)
int workspaceLimit = -1;
int shareWorkspace = 1;
size_t workspaceSize = 0;
void* workspace;


float alpha = 1.0f, beta = 0.0f;

// convert a tensor to a cudnn tensor descriptor
cudnnTensorDescriptor_t cudnn_create_tensor_descriptor(THTensor* t){
	cudnnTensorDescriptor_t tensor;

	checkCUDNN(cudnnCreateTensorDescriptor(&tensor));

	int dim = t->nDimension > 3 ? t->nDimension : 4;

	int size[dim];
	int stride[dim];

	int i;
	if(t->nDimension==1){
		for(i = 0; i< dim; i++){
			size[i] = 1;
			stride[i] = 1;
		}
		size[3] = (int)(t->size[0]);
	} else if(t->nDimension==3){
		// in case of 3d tensor, treat as 3d image of batch 1
		size[0] = 1;
		stride[0] = (int)(t->stride[0]);
		for(i = 0; i< t->nDimension; i++){
			size[i+1] = (int)(t->size[i]);
			stride[i+1] = (int)(t->stride[i]);
		}
	} else {
		for(i = 0; i< dim; i++){
			size[i] = i < t->nDimension ? (int)(t->size[i]) : 1;
			stride[i] = i < t->nDimension ? (int)(t->stride[i]) : 1;
		}
	}

	checkCUDNN(cudnnSetTensorNdDescriptor(tensor, CUDNN_DATA_FLOAT,
				dim, size, stride));

	return tensor;
}

// forward/backward methods for all activations
jobject cudnn_activation_forward(JNIEnv * env, jobject out, jobject in, cudnnActivationMode_t act){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor_(resizeAs)(state, output, input);

	// create cudnn tensor descriptors
	cudnnTensorDescriptor_t inputTensor = cudnn_create_tensor_descriptor(input);
	cudnnTensorDescriptor_t outputTensor = cudnn_create_tensor_descriptor(output);

	// activation descriptor
	cudnnActivationDescriptor_t activation;
	checkCUDNN(cudnnCreateActivationDescriptor(&activation));

	// set activation descriptor
	checkCUDNN(cudnnSetActivationDescriptor(activation,
							act,
							CUDNN_PROPAGATE_NAN, 0.0));

	// do activation
	checkCUDNN(cudnnActivationForward(cudnnHandle, activation,
				&alpha, inputTensor, THTensor_(data)(state, input),
				&beta, outputTensor, THTensor_(data)(state, output)));

	// cleanup cudnn descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
	checkCUDNN(cudnnDestroyActivationDescriptor(activation));

	return out == NULL ? createTensorObject(env, output) : out;
}


jobject cudnn_activation_backward(JNIEnv * env, jobject gradIn, jobject gradOut, jobject in, jobject out, cudnnActivationMode_t act){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor_(resizeAs)(state, input, output);
	THTensor_(resizeAs)(state, gradOutput, output);
	THTensor_(resizeAs)(state, gradInput, output);

	// create cudnn tensor descriptors
	cudnnTensorDescriptor_t inputTensor = cudnn_create_tensor_descriptor(input);
	cudnnTensorDescriptor_t outputTensor = cudnn_create_tensor_descriptor(output);
	cudnnTensorDescriptor_t gradInputTensor = cudnn_create_tensor_descriptor(gradInput);
	cudnnTensorDescriptor_t gradOutputTensor = cudnn_create_tensor_descriptor(gradOutput);

	// activation descriptor
	cudnnActivationDescriptor_t activation;
	checkCUDNN(cudnnCreateActivationDescriptor(&activation));

	// set activation descriptor
	checkCUDNN(cudnnSetActivationDescriptor(activation,
							act,
							CUDNN_PROPAGATE_NAN, 0.0));

	// do activation
	checkCUDNN(cudnnActivationBackward(cudnnHandle, activation,
				&alpha, outputTensor, THTensor_(data)(state, output),
				gradOutputTensor, THTensor_(data)(state, gradOutput),
				inputTensor, THTensor_(data)(state, input),
				&beta, gradInputTensor, THTensor_(data)(state, gradInput)));

	// cleanup cudnn descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradInputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradOutputTensor));
	checkCUDNN(cudnnDestroyActivationDescriptor(activation));

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_tanh
  (JNIEnv * env, jclass c, jobject out, jobject in){
	return cudnn_activation_forward(env, out, in, CUDNN_ACTIVATION_TANH);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_tanhGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject out){
	return cudnn_activation_backward(env, gradIn, gradOut, in, out, CUDNN_ACTIVATION_TANH);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_sigmoid
  (JNIEnv * env, jclass c, jobject out, jobject in){
	return cudnn_activation_forward(env, out, in, CUDNN_ACTIVATION_SIGMOID);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_sigmoidGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject out){
	return cudnn_activation_backward(env, gradIn, gradOut, in, out, CUDNN_ACTIVATION_SIGMOID);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_relu
  (JNIEnv * env, jclass c, jobject out, jobject in){
	return cudnn_activation_forward(env, out, in, CUDNN_ACTIVATION_RELU);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_reluGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject out){
	return cudnn_activation_backward(env, gradIn, gradOut, in, out, CUDNN_ACTIVATION_RELU);
}


// softmax

jobject cudnn_softmax_forward(JNIEnv * env, jobject out, jobject in, cudnnSoftmaxAlgorithm_t sma){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor_(resizeAs)(state, output, input);

	// create cudnn tensor descriptors
	cudnnTensorDescriptor_t inputTensor = cudnn_create_tensor_descriptor(input);
	cudnnTensorDescriptor_t outputTensor = cudnn_create_tensor_descriptor(output);

	// do softmax
	checkCUDNN(cudnnSoftmaxForward(cudnnHandle, sma,
			input->nDimension > 2 ? CUDNN_SOFTMAX_MODE_CHANNEL : CUDNN_SOFTMAX_MODE_INSTANCE,
			&alpha, inputTensor, THTensor_(data)(state, input),
			&beta, outputTensor, THTensor_(data)(state, output)));

	// cleanup cudnn descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));

	return out == NULL ? createTensorObject(env, output) : out;
}


jobject cudnn_softmax_backward(JNIEnv * env, jobject gradIn, jobject gradOut, jobject out, cudnnSoftmaxAlgorithm_t sma){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* output = getTensor(env, out);
	THTensor_(resizeAs)(state, gradInput, gradOutput);

	// create cudnn tensor descriptors
	cudnnTensorDescriptor_t outputTensor = cudnn_create_tensor_descriptor(output);
	cudnnTensorDescriptor_t gradInputTensor = cudnn_create_tensor_descriptor(gradInput);
	cudnnTensorDescriptor_t gradOutputTensor = cudnn_create_tensor_descriptor(gradOutput);

	// do softmax
	checkCUDNN(cudnnSoftmaxBackward(cudnnHandle, sma,
				output->nDimension > 2 ? CUDNN_SOFTMAX_MODE_CHANNEL : CUDNN_SOFTMAX_MODE_INSTANCE,
				&alpha, outputTensor, THTensor_(data)(state, output),
				gradOutputTensor, THTensor_(data)(state, gradOutput),
				&beta, gradInputTensor, THTensor_(data)(state, gradInput)));

	// cleanup cudnn descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradInputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradOutputTensor));

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_softmax
  (JNIEnv * env, jclass c, jobject out, jobject in){
	return cudnn_softmax_forward(env, out, in, CUDNN_SOFTMAX_ACCURATE);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_softmaxGradIn
(JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject out){
	return cudnn_softmax_backward(env, gradIn, gradOut, out, CUDNN_SOFTMAX_ACCURATE);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_logsoftmax
  (JNIEnv * env, jclass c, jobject out, jobject in){
	return cudnn_softmax_forward(env, out, in, CUDNN_SOFTMAX_LOG);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_logsoftmaxGradIn
(JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject out){
	return cudnn_softmax_backward(env, gradIn, gradOut, out, CUDNN_SOFTMAX_LOG);
}


// batch norm
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
	THTensor* running_std = getTensor(env, rv);
	THTensor* save_mean = getTensor(env, sm);
	THTensor* save_std = getTensor(env, sv);


	// create cudnn tensor descriptors
	cudnnTensorDescriptor_t inputTensor;
	cudnnTensorDescriptor_t outputTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));

	// our BN tensors are all setup as nBatches,nFeatures,nData
	// so here 3d to 4d should mean adding a 1 at the back
	int dim = 4;
	int size[dim];
	int stride[dim];
	int i;
	for(i = 0; i< dim; i++){
		size[i] = i < input->nDimension ? (int)(input->size[i]) : 1;
		stride[i] = i < input->nDimension ? (int)(input->stride[i]) : 1;
	}
	checkCUDNN(cudnnSetTensorNdDescriptor(inputTensor, CUDNN_DATA_FLOAT,
				dim, size, stride));
	checkCUDNN(cudnnSetTensorNdDescriptor(outputTensor, CUDNN_DATA_FLOAT,
					dim, size, stride));

	cudnnTensorDescriptor_t scaleBiasMeanVarTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&scaleBiasMeanVarTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(scaleBiasMeanVarTensor,
	                                      CUDNN_TENSOR_NCHW,
	                                      CUDNN_DATA_FLOAT,
	                                      1, running_mean->size[0], 1, 1));

	if(train){
		checkCUDNN(cudnnBatchNormalizationForwardTraining(cudnnHandle,
				input->nDimension > 2 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
				&alpha, &beta,
				inputTensor, THTensor_(data)(state, input),
				outputTensor, THTensor_(data)(state, output),
				scaleBiasMeanVarTensor,
				THTensor_(data)(state, weight),
				THTensor_(data)(state, bias),
				0.1,
				THTensor_(data)(state, running_mean),
				THTensor_(data)(state, running_std),
				1e-5,
				THTensor_(data)(state, save_mean),
				THTensor_(data)(state, save_std)));
	} else {
		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle,
				input->nDimension > 2 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
				&alpha, &beta,
				inputTensor, THTensor_(data)(state, input),
				outputTensor, THTensor_(data)(state, output),
				scaleBiasMeanVarTensor,
				THTensor_(data)(state, weight),
				THTensor_(data)(state, bias),
				THTensor_(data)(state, running_mean),
				THTensor_(data)(state, running_std),
				1e-5));
	}


	// cleanup cudnn descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(scaleBiasMeanVarTensor));


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

	// create cudnn tensor descriptors
	cudnnTensorDescriptor_t inputTensor;
	cudnnTensorDescriptor_t gradInputTensor;
	cudnnTensorDescriptor_t gradOutputTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradInputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradOutputTensor));

	// our BN tensors are all setup as nBatches,nFeatures,nData
	// so here 3d to 4d should mean adding a 1 at the back
	int dim = 4;
	int size[dim];
	int stride[dim];
	int i;
	for(i = 0; i< dim; i++){
		size[i] = i < input->nDimension ? (int)(input->size[i]) : 1;
		stride[i] = i < input->nDimension ? (int)(input->stride[i]) : 1;
	}
	checkCUDNN(cudnnSetTensorNdDescriptor(inputTensor, CUDNN_DATA_FLOAT,
				dim, size, stride));
	checkCUDNN(cudnnSetTensorNdDescriptor(gradInputTensor, CUDNN_DATA_FLOAT,
					dim, size, stride));
	checkCUDNN(cudnnSetTensorNdDescriptor(gradOutputTensor, CUDNN_DATA_FLOAT,
					dim, size, stride));

	cudnnTensorDescriptor_t scaleBiasMeanVarTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&scaleBiasMeanVarTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(scaleBiasMeanVarTensor,
	                                      CUDNN_TENSOR_NCHW,
	                                      CUDNN_DATA_FLOAT,
	                                      1, running_mean->size[0], 1, 1));

	// backward - don't update grad weights here
	checkCUDNN(cudnnBatchNormalizationBackward(cudnnHandle,
			input->nDimension > 2 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
			&alpha, &beta, // update input grad
			&beta, &alpha, // dont update weight grad
			inputTensor, THTensor_(data)(state, input),
			gradOutputTensor, THTensor_(data)(state, gradOutput),
			gradInputTensor, THTensor_(data)(state, gradInput),
			scaleBiasMeanVarTensor,
			THTensor_(data)(state, weight),
			THTensor_(data)(state, weight), // should not be altered
			THTensor_(data)(state, weight), // should not be altered
			1e-5,
			THTensor_(data)(state, save_mean),
			THTensor_(data)(state, save_std)));

	// cleanup cudnn descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradOutputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradInputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(scaleBiasMeanVarTensor));


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

	// create cudnn tensor descriptors
	cudnnTensorDescriptor_t inputTensor;
	cudnnTensorDescriptor_t gradInputTensor;
	cudnnTensorDescriptor_t gradOutputTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradInputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradOutputTensor));

	// our BN tensors are all setup as nBatches,nFeatures,nData
	// so here 3d to 4d should mean adding a 1 at the back
	int dim = 4;
	int size[dim];
	int stride[dim];
	int i;
	for(i = 0; i< dim; i++){
		size[i] = i < input->nDimension ? (int)(input->size[i]) : 1;
		stride[i] = i < input->nDimension ? (int)(input->stride[i]) : 1;
	}
	checkCUDNN(cudnnSetTensorNdDescriptor(inputTensor, CUDNN_DATA_FLOAT,
				dim, size, stride));
	checkCUDNN(cudnnSetTensorNdDescriptor(gradInputTensor, CUDNN_DATA_FLOAT,
					dim, size, stride));
	checkCUDNN(cudnnSetTensorNdDescriptor(gradOutputTensor, CUDNN_DATA_FLOAT,
					dim, size, stride));

	cudnnTensorDescriptor_t scaleBiasMeanVarTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&scaleBiasMeanVarTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(scaleBiasMeanVarTensor,
	                                      CUDNN_TENSOR_NCHW,
	                                      CUDNN_DATA_FLOAT,
	                                      1, running_mean->size[0], 1, 1));

	// backward - don't update input grads
	checkCUDNN(cudnnBatchNormalizationBackward(cudnnHandle,
			input->nDimension > 2 ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
			&beta, &alpha, // dont update input grad
			&alpha, &beta, // update weight grad
			inputTensor, THTensor_(data)(state, input),
			gradOutputTensor, THTensor_(data)(state, gradOutput),
			gradOutputTensor, THTensor_(data)(state, gradOutput), // should not be altered
			scaleBiasMeanVarTensor,
			THTensor_(data)(state, weight),
			THTensor_(data)(state, gradWeight),
			THTensor_(data)(state, gradBias),
			1e-5,
			THTensor_(data)(state, save_mean),
			THTensor_(data)(state, save_std)));

	// cleanup cudnn descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradOutputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(scaleBiasMeanVarTensor));

}



// pooling
jobject cudnn_forward_spatial_pool(JNIEnv* env, jobject out, jobject in, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH, cudnnPoolingMode_t mode){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);

	// create cudnn tensor descriptors
	int n = input->nDimension == 4 ? input->size[0] : 1;
	int c = input->nDimension == 4 ? input->size[1] : input->size[0];
	int h = input->nDimension == 4 ? input->size[2] : input->size[1];
	int w =  input->nDimension == 4 ? input->size[3] : input->size[2];

	// declare all cudnn descriptors
	cudnnTensorDescriptor_t inputTensor;
	cudnnTensorDescriptor_t outputTensor;

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));

    // set tensor descriptors
    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

	// create pooling descriptor
	cudnnPoolingDescriptor_t poolDescriptor;
	checkCUDNN(cudnnCreatePoolingDescriptor(&poolDescriptor));

	checkCUDNN(cudnnSetPooling2dDescriptor(poolDescriptor,
	           	mode,
	            CUDNN_PROPAGATE_NAN,
	            kH, kW,
	            pH, pW,
				dH, dW));

	// set output size
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDescriptor,
	           	inputTensor, &n, &c, &h, &w));

    checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

    if(input->nDimension == 3 && n == 1){
		THTensor_(resize3d)(
			state,
			output, c, h, w);
    } else {
		THTensor_(resize4d)(
			state,
			output, n, c, h, w);
    }

	// run pooling
	checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDescriptor,
			&alpha, inputTensor, THTensor_(data)(state, input),
			&beta, outputTensor, THTensor_(data)(state, output)));

	// cleanup
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
	checkCUDNN(cudnnDestroyPoolingDescriptor(poolDescriptor));

	return out == NULL ? createTensorObject(env, output) : out;
}

jobject cudnn_backward_spatial_pool(JNIEnv* env, jobject gradIn, jobject gradOut, jobject in, jobject out, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH, cudnnPoolingMode_t mode){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor_(resizeAs)(state, gradInput, input);


	// create cudnn tensor descriptors
	cudnnTensorDescriptor_t inputTensor;
	cudnnTensorDescriptor_t outputTensor;
	cudnnTensorDescriptor_t gradInputTensor;
	cudnnTensorDescriptor_t gradOutputTensor;

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradInputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradOutputTensor));

	int n = input->nDimension == 4 ? input->size[0] : 1;
	int c = input->nDimension == 4 ? input->size[1] : input->size[0];
	int h = input->nDimension == 4 ? input->size[2] : input->size[1];
	int w =  input->nDimension == 4 ? input->size[3] : input->size[2];

    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

    checkCUDNN(cudnnSetTensor4dDescriptor(gradInputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

	n = gradOutput->nDimension == 4 ? gradOutput->size[0] : 1;
	c = gradOutput->nDimension == 4 ? gradOutput->size[1] : gradOutput->size[0];
	h = gradOutput->nDimension == 4 ? gradOutput->size[2] : gradOutput->size[1];
	w =  gradOutput->nDimension == 4 ? gradOutput->size[3] : gradOutput->size[2];

    checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

    checkCUDNN(cudnnSetTensor4dDescriptor(gradOutputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

	// create pooling descriptor
	cudnnPoolingDescriptor_t poolDescriptor;
	checkCUDNN(cudnnCreatePoolingDescriptor(&poolDescriptor));

	checkCUDNN(cudnnSetPooling2dDescriptor(poolDescriptor,
	           	mode,
	            CUDNN_PROPAGATE_NAN,
	            kH, kW,
	            pH, pW,
				dH, dW));

	// run pooling
	checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDescriptor,
			&alpha, outputTensor, THTensor_(data)(state, output),
			gradOutputTensor, THTensor_(data)(state, gradOutput),
			inputTensor, THTensor_(data)(state, input),
			&beta, gradInputTensor, THTensor_(data)(state, gradInput)));

	// cleanup
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradInputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradOutputTensor));
	checkCUDNN(cudnnDestroyPoolingDescriptor(poolDescriptor));

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}


JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialmaxpool
  (JNIEnv * env, jclass c, jobject out, jobject in, jobject ind, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	return cudnn_forward_spatial_pool(env, out, in, kW, kH, dW, dH, pW, pH, CUDNN_POOLING_MAX);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialmaxpoolGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject out, jobject ind,
		  jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	return cudnn_backward_spatial_pool(env, gradIn, gradOut, in, out, kW, kH, dW, dH, pW, pH, CUDNN_POOLING_MAX);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialavgpool
  (JNIEnv * env, jclass c, jobject out, jobject in, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH, jboolean ceil, jboolean count_pad){
	return cudnn_forward_spatial_pool(env, out, in, kW, kH, dW, dH, pW, pH,
			count_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
}

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialavgpoolGradIn
  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject in, jobject out,
		  jint kW, jint kH, jint dW, jint dH, jint pW, jint pH, jboolean ceil, jboolean count_pad){
	return cudnn_backward_spatial_pool(env, gradIn, gradOut, in, out, kW, kH, dW, dH, pW, pH,
			count_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
}


// convolutions...
JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolve
  (JNIEnv * env, jclass cl, jobject out, jobject in, jobject ker, jobject b, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, ker);
	THTensor* bias = getTensor(env, b);

//start = clock();

	// get input dimensions
	int n = input->nDimension == 4 ? input->size[0] : 1;
	int c = input->nDimension == 4 ? input->size[1] : input->size[0];
	int h = input->nDimension == 4 ? input->size[2] : input->size[1];
	int w =  input->nDimension == 4 ? input->size[3] : input->size[2];

	// declare all cudnn descriptors
	cudnnFilterDescriptor_t filterDescriptor;
	cudnnConvolutionDescriptor_t convDescriptor;
	cudnnConvolutionFwdAlgo_t algo;

	cudnnTensorDescriptor_t inputTensor;
	cudnnTensorDescriptor_t outputTensor;
	cudnnTensorDescriptor_t biasTensor;

	// create cudnn decriptors
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDescriptor));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDescriptor));

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));


	// set filter descriptor
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDescriptor,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          weight->size[0],
                                          weight->size[1]/(kW*kH),
                                          kH,
										  kW));

    // set convolution descriptor
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDescriptor,
                                          pH, pW,
                                          dH, dW,
                                          1, 1,
										  CUDNN_CROSS_CORRELATION));

    // set tensor descriptors
    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1, weight->size[0],
										  1, 1));

    // find dimension of convolution output and reshape output
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDescriptor,
                                                     inputTensor,
                                                     filterDescriptor,
													 &n, &c, &h, &w));
    if(input->nDimension == 3 && n == 1){
		THTensor_(resize3d)(
			state,
			output, c, h, w);
    } else {
		THTensor_(resize4d)(
			state,
			output, n, c, h, w);
    }

    checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  n, c,
												  h, w));

    // select convolution forward algorithm
    if(convFwAlg == -1){
    	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                   inputTensor,
                                                   filterDescriptor,
                                                   convDescriptor,
                                                   outputTensor,
                                                   workspaceLimit == -1 ? CUDNN_CONVOLUTION_FWD_PREFER_FASTEST :
                                                		   workspaceLimit == 0 ? CUDNN_CONVOLUTION_FWD_NO_WORKSPACE :
                                                				   CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
												   ,
                                                   workspaceLimit,
												   &algo));
    } else {
    	algo = static_cast<cudnnConvolutionFwdAlgo_t>(convFwAlg);;
    }

    // setup workspace
	size_t size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                       inputTensor,
                                                       filterDescriptor,
                                                       convDescriptor,
                                                       outputTensor,
                                                       algo,
													   &size));

//    printf("CONV ALGORITHM %d - WORKSPACE SIZE %d \n", algo, size);

    void* ws;
    if(shareWorkspace > 0){
		// resize shared workspace if required
		if(size > workspaceSize){
			THCudaCheck(cudaFree(workspace));
			THCudaCheck(cudaMalloc(&workspace, size));
			workspaceSize = size;
		}

		ws = workspace;
    } else {
    	// use temp tensor to act as local workspace for this op
		THTensor* workspace = getTensor(env, t1);

		THTensor_(resize1d)(
			state,
			workspace, size);

		ws = THTensor_(data)(state, workspace);
    }

//end = clock();
//cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//printf("CPU TIME PREPARING DESCRIPTORS %f \n",cpu_time_used);


    // do convolution forward
	checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, inputTensor, THTensor_(data)(state, input),
											   filterDescriptor, THTensor_(data)(state, weight),
											   convDescriptor, algo,
											   ws, size, &beta,
	                                           outputTensor, THTensor_(data)(state, output)));

	// add bias
	checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasTensor, THTensor_(data)(state, bias),
								&alpha, outputTensor, THTensor_(data)(state, output)));


	// cleanup descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));

	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDescriptor));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDescriptor));

	return out == NULL ? createTensorObject(env, output) : out;
}



JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveGradIn
  (JNIEnv * env, jclass cl, jobject gradIn, jobject gradOut, jobject ker, jobject in, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradInput = getTensor(env, gradIn);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);
	THTensor* weight = getTensor(env, ker);

	// get gradOut dimensions
	int n = gradOutput->nDimension == 4 ? gradOutput->size[0] : 1;
	int c = gradOutput->nDimension == 4 ? gradOutput->size[1] : gradOutput->size[0];
	int h = gradOutput->nDimension == 4 ? gradOutput->size[2] : gradOutput->size[1];
	int w =  gradOutput->nDimension == 4 ? gradOutput->size[3] : gradOutput->size[2];

	// declare cudnn descriptors
	cudnnFilterDescriptor_t filterDescriptor;
	cudnnConvolutionDescriptor_t convDescriptor;
	cudnnConvolutionBwdDataAlgo_t algo;

	cudnnTensorDescriptor_t gradInputTensor;
	cudnnTensorDescriptor_t gradOutputTensor;

	// create cudnn descriptors
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDescriptor));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDescriptor));

	checkCUDNN(cudnnCreateTensorDescriptor(&gradInputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradOutputTensor));

	// set filter descriptor
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDescriptor,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          weight->size[0],
                                          weight->size[1]/(kW*kH),
                                          kH,
										  kW));

    // set convolution descriptor
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDescriptor,
                                          pH, pW,
                                          dH, dW,
                                          1, 1,
										  CUDNN_CROSS_CORRELATION));

    // set tensor descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(gradOutputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

    // gradIn has same size as input
    n = input->nDimension == 4 ? input->size[0] : 1;
    c = input->nDimension == 4 ? input->size[1] : input->size[0];
    h = input->nDimension == 4 ? input->size[2] : input->size[1];
    w = input->nDimension == 4 ? input->size[3] : input->size[2];

    checkCUDNN(cudnnSetTensor4dDescriptor(gradInputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

    if(input->nDimension == 3 && n == 1){
		THTensor_(resize3d)(
			state,
			gradInput, c, h, w);
    } else {
		THTensor_(resize4d)(
			state,
			gradInput, n, c, h, w);
    }

    // select backward data algorithm
    if(convBwAlg == -1){
    	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
    							filterDescriptor, gradOutputTensor,
								convDescriptor, gradInputTensor,
								CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
								&algo));
    } else {
        algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(convBwAlg);;
    }

    // fix workspace size
	size_t size = 0;
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
    							filterDescriptor, gradOutputTensor,
								convDescriptor, gradInputTensor,
								algo, &size));

    void* ws;
    if(shareWorkspace > 0){
		// resize shared workspace if required
		if(size > workspaceSize){
			THCudaCheck(cudaFree(workspace));
			THCudaCheck(cudaMalloc(&workspace, size));
			workspaceSize = size;
		}

		ws = workspace;
    } else {
    	// use temp tensor to act as local workspace for this op
		THTensor* workspace = getTensor(env, t2);

		THTensor_(resize1d)(
			state,
			workspace, size);

		ws = THTensor_(data)(state, workspace);
    }


    // execute backward
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha,
											   filterDescriptor, THTensor_(data)(state, weight),
											   gradOutputTensor, THTensor_(data)(state, gradOutput),
											   convDescriptor, algo,
											   ws, size, &beta,
	                                           gradInputTensor, THTensor_(data)(state, gradInput)));


	// cleanup cudnn descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(gradInputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradOutputTensor));

	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDescriptor));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDescriptor));

	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
}



JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveAccGrad
  (JNIEnv * env, jclass cl, jobject gradKer, jobject gradB, jobject gradOut, jobject in, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* gradWeight = getTensor(env, gradKer);
	THTensor* gradBias = getTensor(env, gradB);
	THTensor* gradOutput = getTensor(env, gradOut);
	THTensor* input = getTensor(env, in);

	// declare cudnn descriptors
	cudnnFilterDescriptor_t filterDescriptor;
	cudnnConvolutionDescriptor_t convDescriptor;
	cudnnConvolutionBwdFilterAlgo_t algo;

	cudnnTensorDescriptor_t inputTensor;
	cudnnTensorDescriptor_t gradOutputTensor;
	cudnnTensorDescriptor_t gradBiasTensor;

	// create cudnn descriptors
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDescriptor));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDescriptor));

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradOutputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradBiasTensor));

	// set filter descriptor
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDescriptor,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          gradWeight->size[0],
										  gradWeight->size[1]/(kW*kH),
                                          kH,
										  kW));

    // set convolution descriptor
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDescriptor,
                                          pH, pW,
                                          dH, dW,
                                          1, 1,
										  CUDNN_CROSS_CORRELATION));

    // set tensor descriptors
	int n = input->nDimension == 4 ? input->size[0] : 1;
	int c = input->nDimension == 4 ? input->size[1] : input->size[0];
	int h = input->nDimension == 4 ? input->size[2] : input->size[1];
	int w =  input->nDimension == 4 ? input->size[3] : input->size[2];

    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

    n = gradOutput->nDimension == 4 ? gradOutput->size[0] : 1;
    c = gradOutput->nDimension == 4 ? gradOutput->size[1] : gradOutput->size[0];
    h = gradOutput->nDimension == 4 ? gradOutput->size[2] : gradOutput->size[1];
    w = gradOutput->nDimension == 4 ? gradOutput->size[3] : gradOutput->size[2];

    checkCUDNN(cudnnSetTensor4dDescriptor(gradOutputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

    checkCUDNN(cudnnSetTensor4dDescriptor(gradBiasTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1, gradWeight->size[0],
										  1, 1));


    // select backward filter algorithm
    if(convAgAlg == -1) {
    	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
    							inputTensor, gradOutputTensor,
								convDescriptor, filterDescriptor,
								CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
								&algo));
    } else {
    	algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(convFwAlg);
    }

    // fix workspace size
	size_t size = 0;
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
    							inputTensor, gradOutputTensor,
								convDescriptor, filterDescriptor,
								algo, &size));

    void* ws;
    if(shareWorkspace > 0){
		// resize shared workspace if required
		if(size > workspaceSize){
			THCudaCheck(cudaFree(workspace));
			THCudaCheck(cudaMalloc(&workspace, size));
			workspaceSize = size;
		}

		ws = workspace;
    } else {
    	// use temp tensor to act as local workspace for this op
		THTensor* workspace = getTensor(env, t2);

		THTensor_(resize1d)(
			state,
			workspace, size);

		ws = THTensor_(data)(state, workspace);
    }


    // calculate gradient on the bias
    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha,
    									gradOutputTensor, THTensor_(data)(state, gradOutput),
										&beta, gradBiasTensor, THTensor_(data)(state, gradBias)));


    // calculate gradient on the weights
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha,
    									inputTensor, THTensor_(data)(state, input),
										gradOutputTensor, THTensor_(data)(state, gradOutput),
										convDescriptor, algo,
										ws, size, &beta,
										filterDescriptor, THTensor_(data)(state, gradWeight)));

    // cleanup cudnn descriptors
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradOutputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(gradBiasTensor));

	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDescriptor));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDescriptor));
}


