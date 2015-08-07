package be.iminds.iot.dianne.api.nn.module;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * The Output Module is a special Module that marks the end of a neural network chain.
 * 
 * This is the end point of a neural network, which often is a resulting 1D Tensor classifying
 * the input data. The Output can also give access to a number of labels corresponding with the
 * possible output classes.
 * 
 * @author tverbele
 *
 */
public interface Output extends Module {

	/**
	 * Get the last output
	 * @return the output data
	 */
	Tensor getOutput();

	/**
	 * Get the last output tags
	 * @return the tags
	 */
	String[] getTags();
	
	/**
	 * Get the labels of the output this neural network was trained for
	 * @return output labels
	 */
	String[] getOutputLabels();
	
	/**
	 * Set the labels of the output
	 * @param labels new output labels
	 */
	void setOutputLabels(String[] labels);
	
	/**
	 * Initiate a backward pass with a gradient of the output.
	 * 
	 * @param gradOutput the gradient on the output
	 * @param tags optional tags to tag this backward pass
	 */
	void backpropagate(Tensor gradOutput, String... tags);
	
}
