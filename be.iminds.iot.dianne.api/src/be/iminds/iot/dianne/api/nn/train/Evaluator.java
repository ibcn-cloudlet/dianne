package be.iminds.iot.dianne.api.nn.train;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.train.Evaluation;

/**
 * Evaluates a neural network instance given a test Dataset
 * 
 * @author tverbele
 *
 */
public interface Evaluator {

	/**
	 * Evaluate a neural network instance given a test Dataset
	 * 
	 * @param input the Input module of the neural network instance
	 * @param output the Output module of the neural network instance
	 * @param data the test Dataset on which to evaluate
	 * @return the resulting Evaluation object with the resulting confusion matrix
	 */
	Evaluation evaluate(final Input input, final Output output, final Dataset data);

}
