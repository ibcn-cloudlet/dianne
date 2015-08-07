package be.iminds.iot.dianne.api.nn.train.api;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The optimization Criterion for training a neural network.
 * 
 * @author tverbele
 *
 */
public interface Criterion {

	/**
	 * Returns the error metric when comparing the actual output with a target output
	 * 
	 * @param output the neural network output
	 * @param target the desired target output
	 * @return an error metric for the output
	 */
	Tensor forward(final Tensor output, final Tensor target);
	
	/**
	 * Returns gradient feeding into the Output Module backwards for training
	 * 
	 * @param output the neural network output
	 * @param target the desired target output
	 * @return the gradient for backpropagation
	 */
	Tensor backward(final Tensor output, final Tensor target);
	
}
