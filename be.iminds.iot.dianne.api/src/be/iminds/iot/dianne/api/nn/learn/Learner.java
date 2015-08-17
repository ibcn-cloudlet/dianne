package be.iminds.iot.dianne.api.nn.learn;

import java.util.Map;

/**
 * The Learner implements the flow of the learning process. The Learner instantiates
 * a new instance of the neural network to learn, instantiates the Processor to process items
 * from the Dataset, and pushes weight updates to the DianneRepository.
 * 
 * @author tverbele
 *
 */
public interface Learner {

	/**
	 * Start learning for a given neural network and dataset, using the given processor.
	 * The learning will start in a background thread and this method immediately returns.
	 * 
	 * @param nnName the name of the neural network that should be updated
	 * @param dataset the name of the dataset to process for learning
	 * @param config the learner configuration to use
	 */
	void learn(String nnName, String dataset, Map<String, String> config) throws Exception;
	
	/**
	 * Stop the current learning session.
	 */
	void stop();
}
