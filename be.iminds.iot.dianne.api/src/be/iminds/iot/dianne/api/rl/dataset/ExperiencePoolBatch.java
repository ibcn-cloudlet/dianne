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

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A helper class for representing a batch of experience pool samples
 * 
 * @author tverbele
 *
 */
public class ExperiencePoolBatch extends Batch {

	public Tensor reward;
	public Tensor nextState;
	public Tensor terminal;
	
	public ExperiencePoolSample[] samples;
	
	protected ExperiencePoolBatch(){};
	
	public ExperiencePoolBatch(int batchSize, int[] stateDims, int[] actionDims){
		this.input = new Tensor(batchSize, stateDims);
		this.target = new Tensor(batchSize, actionDims);
		this.reward = new Tensor(batchSize, 1);
		this.nextState = new Tensor(batchSize, stateDims);
		this.terminal = new Tensor(batchSize, 1);

		this.samples = new ExperiencePoolSample[batchSize];
		for(int i = 0; i< batchSize; i++){
			this.samples[i] = new ExperiencePoolSample(input.select(0, i), target.select(0, i),
					reward.select(0, i), nextState.select(0, i), terminal.select(0, i));
		}
	}
	
	public ExperiencePoolBatch(Tensor states, Tensor actions, Tensor rewards, Tensor nextStates, Tensor terminal){
		this.input = states;
		this.target = actions;
		this.reward = rewards;
		if(rewards.dim() == 1){
			this.reward.reshape(rewards.size(0), 1);
		}
		this.nextState = nextStates;
		this.terminal = terminal;
		if(terminal.dim() == 1){
			this.terminal.reshape(terminal.size(0), 1);
		}
		
		int batchSize = input.size(0);
		this.samples = new ExperiencePoolSample[batchSize];
		for(int i = 0; i< batchSize; i++){
			this.samples[i] = new ExperiencePoolSample(input.select(0, i), target.select(0, i),
					reward.select(0, i), nextState.select(0, i), terminal.select(0, i));
		}
	}

	public ExperiencePoolBatch(int batchSize){
		// fill with empty samples...
		this.samples = new ExperiencePoolSample[batchSize];
		for(int i = 0; i< batchSize; i++){
			this.samples[i] = new ExperiencePoolSample();
		}
	}
	
	public ExperiencePoolBatch(ExperiencePoolSample[] samples){
		// no single batch Tensor exists
		// only an array of separate tensors
		this.samples = samples;
	}
	
	public int getSize(){
		return samples.length;
	}
	
	public ExperiencePoolSample getSample(int i){
		return samples[i];
	}
	
	public Tensor getState(){
		return input;
	}
	
	public Tensor getAction(){
		return target;
	}
	
	public Tensor getReward(){
		return reward;
	}
	
	public Tensor getNextState(){
		return nextState;
	}
	
	public Tensor getTerminal(){
		return terminal;
	}
	
	public Tensor getInput(int i){
		return samples[i].input;
	}
	
	public Tensor getTarget(int i){
		return samples[i].target;
	}

	public Tensor getState(int i){
		return samples[i].input;
	}
	
	public Tensor getAction(int i){
		return samples[i].target;
	}
	
	public Tensor getReward(int i){
		return samples[i].reward;
	}
	
	public float getScalarReward(int i){
		return samples[i].getScalarReward();
	}
	
	public Tensor getNextState(int i){
		return samples[i].nextState;
	}
	
	public boolean isTerminal(int i){
		return samples[i].isTerminal();
	}
	
	@Override
	public String toString(){
		StringBuilder b = new StringBuilder();
		b.append("State: ")
		.append(input)
		.append(" - Action: ")
		.append(target)
		.append(" - Reward: ")
		.append(reward)
		.append(" - Next state: ")
		.append(nextState)
		.append(" - Terminal: ")
		.append(terminal);
		return b.toString();
	}
}
