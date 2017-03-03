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

import java.io.IOException;

import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.dataset.SequenceDataset;

/**
 * An ExperiencePool is a special kind of SequenceDataset with some extra functionality 
 * specifically for Reinforcement Learning.
 * 
 * Besides the input (= state) and the output (= action) sample, the experience
 * pool also provides the reward and next state for a sample. At runtime a RL Agent
 * can also add new samples to the ExperiencePool
 * 
 * @author tverbele
 *
 */
public interface ExperiencePool extends SequenceDataset<ExperiencePoolSample, ExperiencePoolBatch> {

	/**
	 * Returns the dimensions of the state
	 * 
	 * @return state dimensions
	 */
	default int[] stateDims(){
		return inputDims();
	}
	
	default String stateType(){
		return inputType();
	}
	
	/**
	 * Returns the dimensions of the actions
	 * 
	 * @return action dimensions
	 */
	default int[] actionDims(){
		return targetDims();
	}
	
	default String actionType(){
		return targetType();
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

	default RawExperiencePoolSample getRawSample(final int index){
		ExperiencePoolSample s = getSample(index);
		return new RawExperiencePoolSample(s.input.dims(), s.target.dims(), 
				s.input.get(), s.target.get(), s.nextState.get(), s.reward.get(), s.terminal.get());
	}
	
	/**
	 * Get a batch from the experience pool. 
	 * 
	 * @param indices the indices to fetch, should be smaller then size()
	 * @return the batch containing the requested samples
	 */
	default ExperiencePoolBatch getBatch(final int... indices){
		return getBatch(null, indices);
	}
	
	ExperiencePoolBatch getBatch(ExperiencePoolBatch b, final int...indices);
	
	default RawExperiencePoolBatch getRawBatch(final int... indices){
		ExperiencePoolBatch b = getBatch(indices);
		return new RawExperiencePoolBatch(b.input.dims(), b.target.dims(), 
				b.input.get(), b.target.get(), b.nextState.get(), b.reward.get(), b.terminal.get());
	}
	
	/**
	 * Add a new sequence of interactions to the experience pool
	 * 
	 * @param sequence the sequence of samples to add
	 */
	void addSequence(Sequence<ExperiencePoolSample> sequence);

	
	/**
	 * Remove all samples from the experience pool
	 */
	void reset();
	
	/**
	 * Dump the data of the experience pool to file for later recovery
	 * @throws IOException
	 */
	void dump() throws IOException;
	
}
