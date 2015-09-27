package be.iminds.iot.dianne.api.rl;

import java.util.Map;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;

/**
 * QLearner interface for reinforcement learning with target neural network 
 * 
 * @author tverbele
 *
 */
public interface QLearner {

	/**
	 * Start learning for a given neural network and dataset, using the given processor.
	 * The learning will start in a background thread and this method immediately returns.
	 * 
	 * @param nni the neural network instance that should be updated
	 * @param targeti the target neural network instance for Q learning
	 * @param dataset the name of the dataset to process for learning
	 * @param config the learner configuration to use
	 */
	void learn(NeuralNetworkInstanceDTO nni, NeuralNetworkInstanceDTO targeti, String dataset, Map<String, String> config) throws Exception;
	
	/**
	 * Stop the current learning session.
	 */
	void stop();
}
