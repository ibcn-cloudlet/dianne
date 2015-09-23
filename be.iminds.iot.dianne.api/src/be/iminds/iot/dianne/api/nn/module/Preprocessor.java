package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A Preprocessor is a special Module that performs some preprocessing on the input data.
 * 
 * In order to work correctly, this Module often requires to have access to the complete dataset.
 * For example, a Normalization module will estimate parameters from example inputs in a dataset.
 * 
 * @author tverbele
 *
 */
public interface Preprocessor extends Module {

	/**
	 * Generate the parameters using the given dataset
	 * @param data dataset to use for calculating the parameters
	 */
	void preprocess(Dataset data);

	/**
	 * Get the parameters from this Module
	 * @return current parameters
	 */
	Tensor getParameters();
	
	/**
	 * Manually set the parameters for this Module
	 * @param parameters new parameters
	 */
	void setParameters(final Tensor parameters);
	
	/**
	 * Returns if the data is already preprocessed or if the parameters are set
	 */
	boolean isPreprocessed(); 
}
