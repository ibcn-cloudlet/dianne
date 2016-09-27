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

int conv = -1;
// to be used when we use a shared workspace
int workspaceLimit = -1;
int shareWorkspace = 0;
size_t workspaceSize = 0;
void* workspace;


float alpha = 1.0f, beta = 0.0f;

JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolve
  (JNIEnv * env, jclass cl, jobject out, jobject in, jobject ker, jobject b, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
	THTensor* input = getTensor(env, in);
	THTensor* output = getTensor(env, out);
	THTensor* weight = getTensor(env, ker);
	THTensor* bias = getTensor(env, b);


//start = clock();

	int n = input->nDimension == 4 ? input->size[0] : 1;
	int c = input->nDimension == 4 ? input->size[1] : input->size[0];
	int h = input->nDimension == 4 ? input->size[2] : input->size[1];
	int w =  input->nDimension == 4 ? input->size[3] : input->size[2];

	cudnnFilterDescriptor_t filterDescriptor;
	cudnnConvolutionDescriptor_t convDescriptor;
	cudnnConvolutionFwdAlgo_t convAlgo;

	cudnnTensorDescriptor_t inputTensor;
	cudnnTensorDescriptor_t outputTensor;
	cudnnTensorDescriptor_t weightTensor;
	cudnnTensorDescriptor_t biasTensor;

	checkCUDNN(cudnnCreateFilterDescriptor(&filterDescriptor));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDescriptor));

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&weightTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));


    checkCUDNN(cudnnSetFilter4dDescriptor(filterDescriptor,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          weight->size[0],
                                          weight->size[1]/(kW*kH),
                                          kH,
										  kW));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convDescriptor,
                                          pH, pW,
                                          dH, dW,
                                          1, 1,
										  CUDNN_CROSS_CORRELATION));

    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          n, c, h, w));

    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1, weight->size[0],
										  1, 1));

    // Find dimension of convolution output and reshape output
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

    if(conv == -1){
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
												   &convAlgo));
    } else {
    	convAlgo = static_cast<cudnnConvolutionFwdAlgo_t>(conv);;
    }

	size_t size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                       inputTensor,
                                                       filterDescriptor,
                                                       convDescriptor,
                                                       outputTensor,
                                                       convAlgo,
													   &size));

//    printf("CONV ALGORITHM %d - WORKSPACE SIZE %d \n", convAlgo, size);

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


	checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, inputTensor, THTensor_(data)(state, input),
											   filterDescriptor, THTensor_(data)(state, weight),
											   convDescriptor, convAlgo,
											   ws, size, &beta,
	                                           outputTensor, THTensor_(data)(state, output)));

	checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasTensor, THTensor_(data)(state, bias),
								&alpha, outputTensor, THTensor_(data)(state, output)));


	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(weightTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));

	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDescriptor));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDescriptor));

	return out == NULL ? createTensorObject(env, output) : out;
}

//JNIEXPORT jobject JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveGradIn
//  (JNIEnv * env, jclass c, jobject gradIn, jobject gradOut, jobject ker, jobject in, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
//	THTensor* gradInput = getTensor(env, gradIn);
//	THTensor* gradOutput = getTensor(env, gradOut);
//	THTensor* weight = getTensor(env, ker);
//	THTensor* input = getTensor(env, in);
//	THTensor* temp1 = getTensor(env, t1);
//	THTensor* temp2 = getTensor(env, t2);
//
//	printf("Spatial convolve backward in CUDNN! \n");
//
//	throwException("Not yet implemented");
//
//	return gradIn == NULL ? createTensorObject(env, gradInput) : gradIn;
//}
//
//JNIEXPORT void JNICALL Java_be_iminds_iot_dianne_tensor_ModuleOps_spatialconvolveAccGrad
//  (JNIEnv * env, jclass c, jobject gradKer, jobject gradB, jobject gradOut, jobject in, jobject t1, jobject t2, jint kW, jint kH, jint dW, jint dH, jint pW, jint pH){
//	THTensor* gradWeight = getTensor(env, gradKer);
//	THTensor* gradBias = getTensor(env, gradB);
//	THTensor* gradOutput = getTensor(env, gradOut);
//	THTensor* input = getTensor(env, in);
//	THTensor* temp1 = getTensor(env, t1);
//	THTensor* temp2 = getTensor(env, t2);
//
//	printf("Spatial convolve accgrad in CUDNN! \n");
//
//	throwException("Not yet implemented");
//
//}


