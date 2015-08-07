package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Trainable {

	void accGradParameters();
	
	void zeroGradParameters();
	
	void updateParameters(final float learningRate);
	
	Tensor getGradParameters();
	
	Tensor getParameters();
	
	void setParameters(final Tensor parameters);

}
