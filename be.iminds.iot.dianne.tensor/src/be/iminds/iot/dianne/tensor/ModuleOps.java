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
package be.iminds.iot.dianne.tensor;

public class ModuleOps {
	
	public static native Tensor tanh(Tensor output, final Tensor input);

	public static native Tensor tanhGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output);

	
	public static native Tensor sigmoid(Tensor output, final Tensor input);

	public static native Tensor sigmoidGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output);
	
	
	public static native Tensor softplus(Tensor output, final Tensor input, float beta, float threshold);

	public static native Tensor softplusGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output, float beta, float threshold);
	
	
	public static native Tensor elu(Tensor output, final Tensor input, float alpha, boolean inPlace);

	public static native Tensor eluGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output, float alpha, boolean inPlace);
	
	
	public static native Tensor selu(Tensor output, final Tensor input, float alpha, float lambda);

	public static native Tensor seluGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output, float alpha, float lambda);
	
	
	public static native Tensor threshold(Tensor output, final Tensor input, 
			final float threshold, final float val);

	public static native Tensor thresholdGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output, final float threshold, float val);
	
	
	public static native Tensor relu(Tensor output, final Tensor input);

	public static native Tensor reluGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output);
	
	
	public static native Tensor lrelu(Tensor output, final Tensor input, float leakyness);

	public static native Tensor lreluGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output, float leakyness);
	
		
	public static native Tensor prelu(Tensor output, final Tensor input, 
			final Tensor weight, final int noOutputPlanes);

	public static native Tensor preluGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output, final Tensor weight, final int noOutputPlanes);
	
	public static native void preluAccGrad(final Tensor gradWeight, final Tensor gradOutput, final Tensor input, final Tensor output, final Tensor weight, final int noOutputPlanes);
	

	public static native Tensor softmax(Tensor output, final Tensor input);

	public static native Tensor softmaxGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output);

	public static native Tensor logsoftmax(Tensor output, final Tensor input);

	public static native Tensor logsoftmaxGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output);
	

	public static native Tensor temporalmaxpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int dW);

	public static native Tensor temporalmaxpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output, final Tensor indices,
			final int kW, final int dW);
	

	public static native Tensor spatialmaxpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialmaxpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input,  final Tensor output, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	
	public static native Tensor volumetricmaxpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD, final int padW, final int padH, final int padD);

	public static native Tensor volumetricmaxpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input,  final Tensor output, final Tensor indices,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD, final int padW, final int padH, final int padD);
	
	
	
	public static native Tensor spatialmaxunpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialmaxunpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	
	public static native Tensor volumetricmaxunpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD, final int padW, final int padH, final int padD);

	public static native Tensor volumetricmaxunpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD, final int padW, final int padH, final int padD);
	
	
	
	public static native Tensor spatialavgpool(Tensor output, final Tensor input,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH, boolean ceil, boolean count_pad);

	public static native Tensor spatialavgpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor output,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH, boolean ceil, boolean count_pad);

	
	public static native Tensor volumetricavgpool(Tensor output, final Tensor input,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD);

	public static native Tensor volumetricavgpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input,  final Tensor output,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD);

	
	
	public static native Tensor temporalconvolve(Tensor output, final Tensor input, final Tensor kernels, final Tensor bias, 
			final int kW, final int dW, final int inputFrameSize, final int outputFrameSize);

	public static native Tensor temporalconvolveGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor kernels, final Tensor input, 
			final int kW, final int dW);

	public static native void temporalconvolveAccGrad(final Tensor gradKernels, final Tensor gradBias, final Tensor gradOutput, final Tensor input, 
			final int kW, final int dW);
	
	
	public static native Tensor spatialconvolve(Tensor output, final Tensor input, final Tensor kernels, final Tensor bias, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialconvolveGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor kernels, final Tensor input, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native void spatialconvolveAccGrad(final Tensor gradKernels, final Tensor gradBias, final Tensor gradOutput, final Tensor input, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);
	
	
	public static native Tensor volumetricconvolve(Tensor output, final Tensor input, final Tensor kernels, final Tensor bias, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int kD, final int dW, final int dH, int dD, final int padW, final int padH, final int padD);

	public static native Tensor volumetricconvolveGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor kernels, final Tensor input, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD, final int padW, final int padH, final int padD);

	public static native void volumetricconvolveAccGrad(final Tensor gradKernels, final Tensor gradBias, final Tensor gradOutput, final Tensor input, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD, final int padW, final int padH, final int padD);
	
	
	
	public static native Tensor spatialfullconvolve(Tensor output, final Tensor input, final Tensor kernels, final Tensor bias, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialfullconvolveGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor kernels, final Tensor input, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native void spatialfullconvolveAccGrad(final Tensor gradKernels, final Tensor gradBias, final Tensor gradOutput, final Tensor input, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);
	
	
	public static native Tensor volumetricfullconvolve(Tensor output, final Tensor input, final Tensor kernels, final Tensor bias, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int kD, final int dW, final int dH, int dD, final int padW, final int padH, final int padD);

	public static native Tensor volumetricfullconvolveGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor kernels, final Tensor input, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD, final int padW, final int padH, final int padD);

	public static native void volumetricfullconvolveAccGrad(final Tensor gradKernels, final Tensor gradBias, final Tensor gradOutput, final Tensor input, 
			final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD, final int padW, final int padH, final int padD);
	
	
	
	public static native Tensor batchnorm(Tensor output, final Tensor input, final Tensor weights, final Tensor bias, final Tensor rMean, final Tensor rVar, final Tensor sMean, final Tensor sVar, boolean train);

	public static native Tensor batchnormGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor weights, final Tensor rMean, final Tensor rVar, final Tensor sMean, final Tensor sVar, boolean train);

	public static native void batchnormAccGrad(final Tensor gradWeights, final Tensor gradBias, final Tensor gradOutput, final Tensor input, final Tensor weights, final Tensor rMean, final Tensor rVar, final Tensor sMean, final Tensor sVar, boolean train);

	
	public static native Tensor linear(Tensor output, final Tensor input, final Tensor weights, final Tensor bias, final Tensor ones);
	
	public static native Tensor linearGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor weights, final Tensor input);
	
	public static native void linearAccGrad(final Tensor gradWeigths, final Tensor gradBias, final Tensor gradOutput, final Tensor input, final Tensor ones);

}
