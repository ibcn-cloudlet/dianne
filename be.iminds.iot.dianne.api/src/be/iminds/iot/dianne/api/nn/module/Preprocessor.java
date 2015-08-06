package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

public interface Preprocessor {

	void preprocess(Dataset data);
	
	Tensor getParameters();
	
	void setParameters(final Tensor parameters);
}
