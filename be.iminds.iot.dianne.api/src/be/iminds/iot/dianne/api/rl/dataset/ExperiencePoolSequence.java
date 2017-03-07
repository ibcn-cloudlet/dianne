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

import java.util.ArrayList;
import java.util.List;

import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A helper class for representing a sequence of samples of an experiencepool
 * 
 * @author tverbele
 *
 */
public class ExperiencePoolSequence extends Sequence<ExperiencePoolSample> {
	
	public List<Tensor> getStates(){
		ArrayList<Tensor> states = new ArrayList<>();
		for(ExperiencePoolSample t : this){
			states.add(t.input);
		}
		return states;
	}
	
	public Tensor getState(int index){
		return get(index).input;
	}

	public List<Tensor> getActions(){
		ArrayList<Tensor> actions = new ArrayList<>();
		for(ExperiencePoolSample t : this){
			actions.add(t.target);
		}
		return actions;
	}
	
	public Tensor getAction(int index){
		return get(index).target;
	}
	
	public List<Tensor> getRewards(){
		ArrayList<Tensor> rewards = new ArrayList<>();
		for(ExperiencePoolSample t : this){
			rewards.add(t.reward);
		}
		return rewards;
	}
	
	public Tensor getReward(int index){
		return get(index).reward;
	}
	
	public List<Tensor> getNextStates(){
		ArrayList<Tensor> states = new ArrayList<>();
		for(ExperiencePoolSample t : this){
			states.add(t.nextState);
		}
		return states;
	}
	
	public Tensor getNextState(int index){
		return get(index).nextState;
	}
	
	public List<Tensor> getTerminals(){
		ArrayList<Tensor> terminals = new ArrayList<>();
		for(ExperiencePoolSample t : this){
			terminals.add(t.terminal);
		}
		return terminals;
	}
	
	public Tensor getTerminal(int index){
		return get(index).terminal;
	}
}
