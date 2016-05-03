package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public class ModuleOps {

	public static native Tensor tanh(Tensor output, final Tensor input);

	public static native Tensor tanhDin(Tensor gradInput, final Tensor gradOutput, final Tensor output);

	
	public static native Tensor sigmoid(Tensor output, final Tensor input);

	public static native Tensor sigmoidDin(Tensor gradInput, final Tensor gradOutput, final Tensor output);
	
	
	public static native Tensor threshold(Tensor output, final Tensor input, 
			final float threshold, final float coeff, final float offset);

	public static native Tensor thresholdDin(Tensor gradInput, final Tensor gradOutput, final Tensor input, final float threshold, final float coeff);
	

	public static native Tensor softmax(Tensor output, final Tensor input);

	public static native Tensor softmaxDin(Tensor gradInput, final Tensor gradOutput, final Tensor output);
	

	public static native Tensor spatialmaxpool(Tensor output, final Tensor input, 
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	public static native Tensor spatialmaxpoolDin(Tensor gradInput, final Tensor gradOutput, final Tensor input,
			final int kW, final int kH, final int dW, final int dH, final int padW, final int padH);

	
	public static native Tensor spatialconvolve(Tensor output, final Tensor input, final Tensor kernels, final Tensor bias, 
			final int strideX, final int strideY, final int padX, final int padY);

	public static native Tensor spatialconvolveDin(Tensor gradInput, final Tensor gradOutput, final Tensor kernels, 
			final int strideX, final int strideY, final int padX, final int padY);

	public static native Tensor spatialconvolveDker(Tensor gradKer, final Tensor add, final Tensor gradOutput, final Tensor input,
			final int strideX, final int strideY, final int padX, final int padY);
	
	public static native Tensor spatialconvolveDbias(Tensor gradBias, final Tensor gradOutput);
	
}
