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
package be.iminds.iot.dianne.rl.experience.adapters;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(
		service={Dataset.class, ExperiencePool.class},	
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.adapters.GoalAdapter")
public class GoalAdapter extends AbstractExperiencePoolAdapter {

	private Random random = new Random(System.currentTimeMillis());
	
	// the actual state size
	public int stateSize;
	
	// the size of the goal vector
	public int goalSize;
	
	// goalsamples = 0, only add final 
	public int goalSamples = 4;
	
	// store states with total discounted reward given the sequence
	public float discount = 0.0f;
	
	@Override
	protected void configure(Map<String, Object> properties) {
		if(properties.containsKey("stateSize")){
			this.stateSize = Integer.parseInt(properties.get("stateSize").toString());
		} else {
			throw new RuntimeException("GoalAdapter needs a stateSize");
		}
		
		if(properties.containsKey("goalSize")){
			this.goalSize = Integer.parseInt(properties.get("goalSize").toString());
		} else {
			throw new RuntimeException("GoalAdapter needs a goalSize");
		}
		
		if(properties.containsKey("goalSamples")){
			this.goalSamples = Integer.parseInt(properties.get("goalSamples").toString());
		} 
		
		if(properties.containsKey("discount")){
			this.discount = Float.parseFloat(properties.get("discount").toString());
		} 
	}

	@Override
	public void addSequence(Sequence<ExperiencePoolSample> sequence){
		// add sequence with initial goal
		List<ExperiencePoolSample> l = new ArrayList<>();
		for(int i=0;i<sequence.size;i++){
			ExperiencePoolSample s = sequence.get(i);
			ExperiencePoolSample ss = s.copyInto(null);
			ss.input = ss.input.narrow(0, stateSize+goalSize);
			ss.nextState = ss.nextState == null ? null : ss.nextState.narrow(0, stateSize+goalSize);
			l.add(ss);
		}
		
		if(discount > 0){
			float reward = 0;
			for(int i=l.size()-1;i>=0;i--){
				ExperiencePoolSample ss = l.get(i);
				reward = reward*discount + ss.getScalarReward();
				ss.reward.set(reward, 0);
			}
		}
		
		Sequence<ExperiencePoolSample> original = new Sequence<>(l, l.size());
		pool.addSequence(original);
		
		// add sequence with end-state goal if k==0
		if(goalSamples == 0){
			Tensor endGoal = sequence.get(sequence.size-1).input.narrow(stateSize+goalSize, goalSize).copyInto(null);
			l.clear();
			for(int i=0;i<sequence.size;i++){
				ExperiencePoolSample s = sequence.get(i);
				ExperiencePoolSample ss = s.copyInto(null);
				ss.input = ss.input.narrow(0, stateSize+goalSize);
				endGoal.copyInto(ss.input.narrow(stateSize, goalSize));
				ss.nextState = ss.nextState.narrow(0, stateSize+goalSize);
				endGoal.copyInto(ss.nextState.narrow(stateSize, goalSize));
				l.add(ss);
			}
			
			if(discount > 0){
				float reward = 0;
				for(int i=l.size()-1;i>=0;i--){
					ExperiencePoolSample ss = l.get(i);
					reward = reward*discount + ss.getScalarReward();
					ss.reward.set(reward, 0);
				}
			}
			
			Sequence<ExperiencePoolSample> endState = new Sequence<>(l, l.size());
			pool.addSequence(endState);
		} else {	
			// add sequences with intermediate state goals
			for(int k=0;k<goalSamples;k++){
				int sample = 1+random.nextInt(sequence.size-1);
				Tensor endGoal = sequence.get(sample).input.narrow(stateSize+goalSize, goalSize).copyInto(null);
				l.clear();
				for(int i=0;i<sample;i++){
					ExperiencePoolSample s = sequence.get(i);
					ExperiencePoolSample ss = s.copyInto(null);
					ss.input = ss.input.narrow(0, stateSize+goalSize);
					endGoal.copyInto(ss.input.narrow(stateSize, goalSize));
					ss.nextState = ss.nextState.narrow(0, stateSize+goalSize);
					endGoal.copyInto(ss.nextState.narrow(stateSize, goalSize));
					l.add(ss);
				}
				l.get(l.size()-1).nextState = null;
				l.get(l.size()-1).terminal.set(0.0f, 0);
				l.get(l.size()-1).reward.set(0.0f, 0);
				
				if(discount > 0){
					float reward = 0;
					for(int i=l.size()-1;i>=0;i--){
						ExperiencePoolSample ss = l.get(i);
						reward = reward*discount + ss.getScalarReward();
						ss.reward.set(reward, 0);
					}
				}
				
				Sequence<ExperiencePoolSample> endState = new Sequence<>(l, l.size());
				pool.addSequence(endState);
			}
		}
	}
	
}
