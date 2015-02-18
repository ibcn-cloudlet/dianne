package be.iminds.iot.dianne.nn.module;

import be.iminds.iot.dianne.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

public interface Preprocessor {

	public void preprocess(Dataset data);
	
	public Tensor getParameters();
	
	public void setParameters(float[] weigths);
}
