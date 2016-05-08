package be.iminds.iot.dianne.tensor;

public class ModuleOps {
	
	public static native Tensor tanh(Tensor output, final Tensor input);

	public static native Tensor tanhGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor output);

	
	public static native Tensor sigmoid(Tensor output, final Tensor input);

	public static native Tensor sigmoidGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor output);
	
	
	public static native Tensor threshold(Tensor output, final Tensor input, 
			final float threshold, final float val);

	public static native Tensor thresholdGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final float threshold);
	

	public static native Tensor softmax(Tensor output, final Tensor input);

	public static native Tensor softmaxGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor output);
	

	public static native Tensor spatialmaxpool(Tensor output, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialmaxpoolGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor input, final Tensor indices,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	
	public static native Tensor spatialconvolve(Tensor output, final Tensor input, final Tensor kernels, final Tensor bias, final Tensor finput,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialconvolveGradIn(Tensor gradInput, final Tensor gradOutput, final Tensor kernels, final Tensor input, final Tensor finput, final Tensor fgradInput,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native void spatialconvolveAccGrad(final Tensor gradKernels, final Tensor gradBias, final Tensor gradOutput, final Tensor input, final Tensor finput,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);
}
