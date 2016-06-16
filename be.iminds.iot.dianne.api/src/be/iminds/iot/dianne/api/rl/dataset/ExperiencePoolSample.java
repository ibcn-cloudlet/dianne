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

import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A helper class for representing one sample of an experience pool
 * 
 * @author tverbele
 *
 */
public class ExperiencePoolSample extends Sample {

	public float reward;
	public Tensor nextState;
	// terminal states can be marked by the Environment nextState == null or a nextState = -Float.MAX_VALUE
	// use additional boolean to flag this, which allows the ExperiencePoolSample to be reused
	// even if nextState is set to null by the env
	public boolean isTerminal;
	
	public ExperiencePoolSample(){}
	
	public ExperiencePoolSample(Tensor state, Tensor action, float reward, Tensor nextState){
		super(state, action);
		this.reward = reward;
		this.nextState = nextState;
		this.isTerminal = nextState == null;
	}
	
	public Tensor getState(){
		return input;
	}
	
	public Tensor getAction(){
		return target;
	}
	
	public float getReward(){
		return reward;
	}
	
	public Tensor getNextState(){
		return nextState;
	}
	
	public boolean isTerminal(){
		return isTerminal;
	}
}
