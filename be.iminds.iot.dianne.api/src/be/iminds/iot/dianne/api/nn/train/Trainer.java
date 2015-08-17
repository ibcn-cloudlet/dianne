package be.iminds.iot.dianne.api.nn.train;

import java.util.List;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.api.nn.module.Preprocessor;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.nn.train.Criterion;

/**
 * Trains a neural network instance given a training Dataset
 * 
 * @author tverbele
 *
 */
public interface Trainer {

	/**
	 * Trains a neural network instance with a given train Dataset and Criterion
	 * @param input the Input module of the neural network instance to train
	 * @param output the Output module of the neural network instance to train
	 * @param module the list of Trainable modules of the neural network instance to train
	 * @param preprocessors the list of Preprocessor modules of the neural network instance to configure
	 * @param criterion the Criterion to calculate the error metric
	 * @param data the train Dataset
	 */
	void train(final Input input, final Output output, 
			final List<Trainable> module, final List<Preprocessor> preprocessors, 
			final Criterion criterion, final Dataset data);
	
}
