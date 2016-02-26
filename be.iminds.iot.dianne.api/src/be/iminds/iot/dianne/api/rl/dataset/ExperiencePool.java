/*******************************************************************************
 * DIANNE  - Framework for distributed artificial neural networks
 * Copyright (C) 2015  iMinds - IBCN - UGent
 *
 * This file is part of DIANNE.
 *
 * DIANNE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors:
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.api.rl.dataset;

import java.util.Collection;

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
	 * Get a sample from the experience pool. 
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return the sample at position index
	 */
	ExperiencePoolSample getSample(final int index);

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
	
	/**
	 * Add a collection of samples to the experience pool
	 * 
	 * @param samples
	 */
	void addSamples(Collection<ExperiencePoolSample> samples);
	
	/**
	 * Remove all samples from the experience pool
	 */
	void reset();
	
	/**
	 * Set a maximum size for this Experience Pool, from then on old samples will be replaced
	 * by addSample(s) following a FIFO strategy.
	 * 
	 * @param size the new maximum size
	 */
	void setMaxSize(int size);
}
