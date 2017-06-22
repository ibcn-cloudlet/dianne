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
package be.iminds.iot.dianne.api.nn.learn;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Represents the progress made by a Learner
 * 
 * @author tverbele
 *
 */
public class LearnProgress {

	/** The number of iterations (=number of batches) processed */
	public long iteration;
	
	/** The current minibatch loss perceived by the Learner */
	public float minibatchLoss;
	
	/** Any additional metrics to stuff into the LearnProgress */
	public Map<String, Float> extra;
	
	public LearnProgress(long iteration, float loss){
		this.iteration = iteration;
		this.minibatchLoss = loss;
		this.extra = Collections.unmodifiableMap(new HashMap<>());
	}
	
	public LearnProgress(long iteration, float loss, String[] labels, float[] values){
		this(iteration, loss);
		assert labels.length == values.length;
		Map<String, Float> map = new HashMap<>();
		for(int i=0;i<labels.length;i++){
			map.put(labels[i], values[i]);
		}
		extra = Collections.unmodifiableMap(map);
	}
	
	@Override
	public String toString(){
		StringBuilder builder = new StringBuilder(); 
		builder.append("[LEARNER] Iteration: ").append(iteration)
				.append(" Loss: ").append(minibatchLoss).append(" ");
		extra.entrySet().forEach(e -> builder.append(e.getKey()).append(": ").append(e.getValue()).append(" "));
		return builder.toString();
	}
}
