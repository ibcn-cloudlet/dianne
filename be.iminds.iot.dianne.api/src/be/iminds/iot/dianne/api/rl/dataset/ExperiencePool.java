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

import be.iminds.iot.dianne.api.dataset.Batch;
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
	 * Returns the dimensions of the state
	 * 
	 * @return state dimensions
	 */
	default int[] stateDims(){
		return inputDims();
	}
	
	/**
	 * Returns the dimensions of the actions
	 * 
	 * @return action dimensions
	 */
	default int[] actionDims(){
		return targetDims();
	}
	
	/**
	 * Get a sample from the experience pool. 
	 * 
	 * @param index the index to fetch, should be smaller then size()
	 * @return the sample at position index
	 */
	default ExperiencePoolSample getSample(final int index){
		return getSample(null, index);
	}
	
	ExperiencePoolSample getSample(ExperiencePoolSample s, final int index);

	default Batch getBatch(Batch b, final int...indices){
		throw new UnsupportedOperationException("Batches not (yet) supported for Experience Pools");
	}
	
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
	 * Add a new sample to the experience pool
	 * 
	 * @param state the initial state of the environment
	 * @param action the action done in state
	 * @param reward the reward after doing the action in state
	 * @param nextState the next state of the environment after executing the action
	 */
	void addSample(Tensor state, Tensor action, float reward, Tensor nextState);

	void addSample(ExperiencePoolSample sample);

	
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
	
}
