package be.iminds.iot.dianne.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Dataset {
	
	public int size();
	
	public int inputSize();
	
	public int outputSize();

	public Tensor getInputSample(final int index);
		
	public Tensor getOutputSample(final int index);
		
}
