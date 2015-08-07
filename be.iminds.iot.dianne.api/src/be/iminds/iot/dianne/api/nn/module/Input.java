package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The Input Module is a special Module that only has a next Module, 
 * but no previous Module. This module represents the starting point of a 
 * neural network. 
 * 
 * @author tverbele
 *
 */
public interface Input extends Module {

	/**
	 * The input to send through the neural network 
	 * @param input input data
	 * @param tags optional tags to identify this input throughout the neural network
	 */
	void input(final Tensor input, final String... tags);
	
}
