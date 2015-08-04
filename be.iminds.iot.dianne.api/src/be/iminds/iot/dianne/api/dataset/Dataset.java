package be.iminds.iot.dianne.api.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Dataset {
	
	int size();
	
	int inputSize();
	
	int outputSize();

	Tensor getInputSample(final int index);
		
	Tensor getOutputSample(final int index);
	
	String getName();
	
	String[] getLabels();
}
