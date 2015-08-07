package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The Trainable interface is implemented by Modules that can be trained.
 * 
 * It provides access to the module parameters (often denoted as weights in neural networks),
 * as well as the last calculated gradient on the parameters.
 * 
 * @author tverbele
 *
 */
public interface Trainable {

	/**
	 * Accumulate the gradient on the parameters
	 */
	void accGradParameters();
	
	/**
	 * Reset the gradient on the parameters to zero
	 */
	void zeroGradParameters();
	
	/**
	 * Perform a parameter update with the current gradient and provided learning rate
	 * 
	 * Useful for simple gradient descent training procedures
	 *  
	 * @param learningRate a scale factor for gradParameters
	 */
	void updateParameters(final float learningRate);
	
	/**
	 * Return the current gradient on the parameters
	 * 
	 * Attention: at the moment this returns a reference to the gradient of the parameters,
	 * only use and change if you know what you are doing.
	 * 
	 * @return the gradient on the parameters
	 */
	Tensor getGradParameters();
	
	/**
	 * Return the current parameters
	 * 
	 * Attention: at the moment this returns a reference to the parameters,
	 * only use and change if you know what you are doing.
	 * 
	 * @return the parameters
	 */
	Tensor getParameters();
	
	/**
	 * Set new parameters for this Module. These parameters are copied.
	 * @param parameters new parameters
	 */
	void setParameters(final Tensor parameters);

}
