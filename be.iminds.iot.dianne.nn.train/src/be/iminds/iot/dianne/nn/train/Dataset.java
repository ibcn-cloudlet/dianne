package be.iminds.iot.dianne.nn.train;

import be.iminds.iot.dianne.tensor.Tensor;

public interface Dataset {
	
	public int size();
	
	public int inputSize();
	
	public int outputSize();

	public Tensor getInputSample(final int index);
	
	public Tensor getInputBatch(final int startIndex, final int size);
	
	public Tensor getOutputSample(final int index);
	
	public Tensor getOutputBatch(final int startIndex, final int size);
	
}
