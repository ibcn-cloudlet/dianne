package be.iminds.iot.dianne.api.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A Dataset is a collection of input data and the corresponding output classes.
 * 
 * The input tensor can be an n-dimensional input (likely 1-D (sequence) or 2-D (images))
 * 
 * The output tensor is a 1 dimensional vector filled with zeros and only 1 for the classes
 * that are represented by the corresponding input sample. 
 * 
 * Each item from the output tensor corresponds with a human-readable label.
 * 
 * A Dataset can be used for supervised training of a neural network.
 * 
 * @author tverbele
 *
 */
public interface Dataset {
	
	/**
	 * Returns the number of input-output samples in the dataset
	 * 
	 * @return the number of input-output samples in the dataset
	 */
	int size();
	
	/**
	 * Returns the total size of an input sample. In case of multi-dimensional
	 * input this is the total size if the Tensor were transformed to one dimension.
	 * Returns -1 if the size of input samples are variable.
	 * 
	 * @return size of an input sample or -1 if variable
	 */
	// TODO should we return dim[] instead for multi-dimensional inputs?
	int inputSize();
	
	/**
	 * Returns the total size of an output vector. Corresponds with the number 
	 * of classes that this dataset identifies.
	 * 
	 * @return size of an output vector
	 */
	int outputSize();

	/**
	 * Get an input sample from the dataset
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return the input sample at position index
	 */
	Tensor getInputSample(final int index);
		
	/**
	 * Get an output vector from the dataset
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return the output vector corresponding with input sample index
	 */
	Tensor getOutputSample(final int index);
	
	/**
	 * A human-readable name for this dataset
	 * 
	 * @return dataset name
	 */
	String getName();
	
	/**
	 * Get human-readable names for the classes represented in an output vector
	 * 
	 * @return human-readable dataset labels
	 */
	String[] getLabels();
}
