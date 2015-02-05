package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Trainable {

	public void accGradParameters();
	
	public void zeroGradParameters();
	
	public void updateParameters(final float learningRate);
	
	public Tensor getParameters();
	
	public void setParameters(Tensor weights);
	
}
