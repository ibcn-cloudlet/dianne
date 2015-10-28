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
public interface Trainable extends Module {

	/**
	 * Accumulate the gradient on the parameters
	 */
	void accGradParameters();
	
	/**
	 * Reset the delta on the parameters to zero
	 */
	void zeroDeltaParameters();

	/**
	 * Add the current delta to the parameters
	 */
	void updateParameters();
	
	/**
	 * Add the current delta to the parameters, scaled by factor scale
	 * 
	 * Useful for simple gradient descent training procedures 
	 * (use negative scale for gradient descent, positive scale for gradient ascent)
	 *  
	 * @param scale a scale factor for delta parameters
	 */
	void updateParameters(final float scale);
	
	/**
	 * Return the current delta on the parameters. This can be accumulated gradients 
	 * by calls to accGradParameters, and/or some custom operations on delta parameters.
	 * 
	 * Attention: at the moment this returns a reference to the gradient of the parameters,
	 * only use and change if you know what you are doing.
	 * 
	 * @return the delta on the parameters
	 */
	Tensor getDeltaParameters();
	
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
	
	/**
	 * Set parameters with random values
	 */
	void randomize();

}
