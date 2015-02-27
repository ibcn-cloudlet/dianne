package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Trainable {

	public void accumulateGradParameters();
	
	public void zeroGradParameters();
	
	public void updateParameters(final float learningRate);
	
	public Tensor getParameters();
	
	public void setParameters(float[] weights);

	public void setFixed(boolean fixed);
}
