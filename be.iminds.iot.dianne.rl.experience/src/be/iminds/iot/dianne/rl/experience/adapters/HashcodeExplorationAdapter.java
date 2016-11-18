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

import java.util.HashMap;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(
		service={Dataset.class, ExperiencePool.class},	
		configurationPolicy=ConfigurationPolicy.REQUIRE,
		configurationPid="be.iminds.iot.dianne.dataset.adapters.HashcodeExplorationAdapter")
public class HashcodeExplorationAdapter extends AbstractExperiencePoolAdapter {

	private Map<Integer, Integer> counts = new HashMap<>();
	
	private float exploration = 1;
	
	@Override
	protected void configure(Map<String, Object> properties) {
		// TODO support different kinds of hashcode?
		
		if(properties.containsKey("exploration")){
			this.exploration = Float.parseFloat(properties.get("exploration").toString());
		}
	}

	@Override
	protected void adaptFetchedSample(ExperiencePoolSample s) {
		Tensor reward = s.getReward();
		Integer count = counts.get(hashcode(s.getState()));
		if(count == null || count == 0){ // TODO should not happen?
			count = 1;
		}
		reward.set(reward.get(0)+exploration/count, 0);
	}

	@Override
	protected void adaptAddingSample(ExperiencePoolSample s) {
		int hashcode = hashcode(s.getState());
		if(!counts.containsKey(hashcode)){
			counts.put(hashcode, 0);
		}
		counts.put(hashcode, counts.get(hashcode) + 1);
	}

	private int hashcode(Tensor state){
		// for now just take the hashcode from toString cast to byte
		// better approaches could be taken?!
		return (byte)state.toString().hashCode();
	}

}
