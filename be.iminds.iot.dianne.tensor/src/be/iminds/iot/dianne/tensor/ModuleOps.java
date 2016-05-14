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
	

	public static native Tensor spatialmaxpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialmaxpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	
	public static native Tensor spatialconvolve(Tensor output, final Tensor input, final Tensor kernels, final Tensor bias, final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialconvolveGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor kernels, final Tensor input, final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native void spatialconvolveAccGrad(final Tensor gradKernels, final Tensor gradBias, final Tensor gradOutput, final Tensor input, final Tensor temp1, final Tensor temp2,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);
	
	
	public static native Tensor batchnorm(Tensor output, final Tensor input, final Tensor weights, final Tensor bias, final Tensor rMean, final Tensor rVar, final Tensor sMean, final Tensor sVar, boolean train);

	public static native Tensor batchnormGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor weights, final Tensor rMean, final Tensor rVar, final Tensor sMean, final Tensor sVar, boolean train);

	public static native void batchnormAccGrad(final Tensor gradWeights, final Tensor gradBias, final Tensor gradOutput, final Tensor input, final Tensor rMean, final Tensor rVar, final Tensor sMean, final Tensor sVar, boolean train);

}
