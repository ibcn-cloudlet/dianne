package be.iminds.iot.dianne.api.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Dataset {
	
	public int size();
	
	public int inputSize();
	
	public int outputSize();

	public Tensor getInputSample(final int index);
		
	public Tensor getOutputSample(final int index);
	
	public String getName();
	
	public String[] getLabels();
}
