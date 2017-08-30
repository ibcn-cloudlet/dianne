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

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;


/**
 * Experience adapter to use as baseline for goal based environments
 * 
 * Instead of storing the goals, it just keeps the original state and 
 * a shaped reward based on the (euclidian) distance between the actual
 * goal vector and the current goal the state is in.
 *
 */
@Component(
		service={Dataset.class, ExperiencePool.class},	
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.adapters.GoalDistanceAdapter")
public class GoalDistanceAdapter extends AbstractExperiencePoolAdapter {

	// the actual state size
	public int stateSize;
	
	// the size of the goal vector
	public int goalSize;
	
	@Override
	protected void configure(Map<String, Object> properties) {
		if(properties.containsKey("stateSize")){
			this.stateSize = Integer.parseInt(properties.get("stateSize").toString());
		} else {
			throw new RuntimeException("GoalDistanceAdapter needs a stateSize");
		}
		
		if(properties.containsKey("goalSize")){
			this.goalSize = Integer.parseInt(properties.get("goalSize").toString());
		} else {
			throw new RuntimeException("GoalDistanceAdapter needs a goalSize");
		}
	}

	@Override
	public void addSequence(Sequence<ExperiencePoolSample> sequence){
		List<ExperiencePoolSample> l = new ArrayList<>();
		for(int i=0;i<sequence.size;i++){
			ExperiencePoolSample s = sequence.get(i);
			ExperiencePoolSample ss = s.copyInto(null);
			Tensor goal = ss.input.narrow(0, stateSize, goalSize);
			Tensor actual = ss.input.narrow(0, stateSize+goalSize, goalSize);
			Tensor diff = TensorOps.sub(actual, actual, goal);
			float sqdiff = TensorOps.sum(TensorOps.cmul(diff, diff, diff));
			
			ss.input = ss.input.narrow(0, stateSize);
			ss.reward.set(-(float)Math.sqrt(sqdiff), 0);
			ss.nextState = ss.nextState == null ? null : ss.nextState.narrow(0, stateSize);
			
			l.add(ss);
		}
		
		Sequence<ExperiencePoolSample> adapted = new Sequence<>(l, l.size());
		pool.addSequence(adapted);
	}
	
}
