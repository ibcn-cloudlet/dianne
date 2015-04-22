package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

public interface Preprocessor {

	public void preprocess(Dataset data);
	
	public Tensor getParameters();
	
	public void setParameters(float[] weigths);
}
