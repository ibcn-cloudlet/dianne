package be.iminds.iot.dianne.tensor;

public class ModuleOps {
	
	public static native Tensor tanh(Tensor output, final Tensor input);

	public static native Tensor tanhGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor output);

	
	public static native Tensor sigmoid(Tensor output, final Tensor input);

	public static native Tensor sigmoidGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor output);
	
	
	public static native Tensor threshold(Tensor output, final Tensor input, 
			final float threshold, final float val);

	public static native Tensor thresholdGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final float threshold);
	
	
	public static native Tensor prelu(Tensor output, final Tensor input, 
			final Tensor weight, final int noOutputPlanes);

	public static native Tensor preluGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor weight, final int noOutputPlanes);
	
	public static native void preluAccGrad(final Tensor gradWeight, final Tensor gradOutput, final Tensor input, final Tensor weight, final int noOutputPlanes);
	

	public static native Tensor softmax(Tensor output, final Tensor input);

	public static native Tensor softmaxGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor output);

	public static native Tensor logsoftmax(Tensor output, final Tensor input);

	public static native Tensor logsoftmaxGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor output);
	

	public static native Tensor temporalmaxpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int dW);

	public static native Tensor temporalmaxpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor indices,
			final int kW, final int dW);
	

	public static native Tensor spatialmaxpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialmaxpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	
	public static native Tensor volumetricmaxpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD, final int padW, final int padH, final int padD);

	public static native Tensor volumetricmaxpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor indices,
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

	public static native Tensor spatialavgpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH, boolean ceil, boolean count_pad);

	
	public static native Tensor volumetricavgpool(Tensor output, final Tensor input,
			final int kW, final int kH, final int kD, final int dW, final int dH, final int dD);

	public static native Tensor volumetricavgpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input,
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
	
	
	
	public static native Tensor batchnorm(Tensor output, final Tensor input, final Tensor weights, final Tensor bias, final Tensor rMean, final Tensor rVar, final Tensor sMean, final Tensor sVar, boolean train);

	public static native Tensor batchnormGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor weights, final Tensor rMean, final Tensor rVar, final Tensor sMean, final Tensor sVar, boolean train);

	public static native void batchnormAccGrad(final Tensor gradWeights, final Tensor gradBias, final Tensor gradOutput, final Tensor input, final Tensor weights, final Tensor rMean, final Tensor rVar, final Tensor sMean, final Tensor sVar, boolean train);

	
	public static native Tensor linear(Tensor output, final Tensor input, final Tensor weights, final Tensor bias);
	
	public static native Tensor linearGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor weights, final Tensor input);
	
	public static native void linearAccGrad(final Tensor gradWeigths, final Tensor gradBias, final Tensor gradOutput, final Tensor input);

}
