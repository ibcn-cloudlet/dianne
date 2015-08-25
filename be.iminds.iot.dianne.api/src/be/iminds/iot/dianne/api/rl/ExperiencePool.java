package be.iminds.iot.dianne.api.rl;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * An ExperiencePool is a special kind of Dataset with some extra functionality 
 * specifically for Reinforcement Learning.
 * 
 * Besides the input (= state) and the output (= action) sample, the experience
 * pool also provides the reward and next state for a sample. At runtime a RL Agent
 * can also add new samples to the ExperiencePool
 * 
 * @author tverbele
 *
 */
public interface ExperiencePool extends Dataset {

	/**
	 * Returns the number of samples in the experience pool
	 * 
	 * @return the number of samples in the experience pool
	 */
	int size();
	
	/**
	 * Returns the total size of an input sample. 
	 * 
	 * @return size of an input sample or -1 if variable
	 */
	// TODO should we return dim[] instead for multi-dimensional inputs?
	int inputSize();
	
	/**
	 * Returns the total size of an output vector. Corresponds with the action dimension
	 * 
	 * @return size of an output vector
	 */
	int outputSize();

	/**
	 * Get an input sample from the experience pool. This vector represents an Environment state
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return the input sample at position index
	 */
	Tensor getInputSample(final int index);
	
	/**
	 * Get the Environment state of a sample, identical to getInputSample()
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return the state of sample at position index
	 */
	Tensor getState(final int index);
		
	/**
	 * Get an output vector from the experience pool. This vector represents the action done in the 
	 * state of this sample
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return the output vector corresponding with input sample index
	 */
	Tensor getOutputSample(final int index);
	
	/**
	 * Get the action done in state of a sample, identical to getOutputSample()
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return the action of sample at position index
	 */
	Tensor getAction(final int index);
	
	/**
	 * A human-readable name for this experience pool.
	 * 
	 * @return dataset name
	 */
	String getName();
	
	/**
	 * Get human-readable names for the actions represented in an output vector
	 * 
	 * @return human-readable action labels
	 */
	String[] getLabels();
	
	/**
	 * Get the reward that the action for this sample resulted in
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return reward
	 */
	float getReward(final int index);
	
	/**
	 * Get the next state for a given sample
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return reward
	 */
	Tensor getNextState(final int index);
	
	/**
	 * Add a new sample to the experience pool
	 * 
	 * @param state the initial state of the environment
	 * @param action the action done in state
	 * @param reward the reward after doing the action in state
	 * @param nextState the next state of the environment after executing the action
	 */
	void addSample(Tensor state, Tensor action, float reward, Tensor nextState);
}
