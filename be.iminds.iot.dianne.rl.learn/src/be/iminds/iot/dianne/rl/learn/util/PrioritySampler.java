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
package be.iminds.iot.dianne.rl.learn.util;

import java.util.Random;
import java.util.TreeMap;

import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Utility class that maintains a prioritized subset of the dataset out of which
 * is sampled for the batches
 * 
 * @author tverbele
 *
 */
public class PrioritySampler {

	private final ExperiencePool pool;
	private final SamplingStrategy sampling;
	private final TreeMap<Float, ExperiencePoolSample> priorityBuffer = new TreeMap<>();
	private final PrioritySamplerConfig config;
	private final Random random = new Random(System.currentTimeMillis());
	
	public PrioritySampler(ExperiencePool pool, SamplingStrategy sampling, PrioritySamplerConfig config){
		this.pool = pool;
		this.sampling = sampling;
		this.config = config;
	}
	
	public ExperiencePoolBatch getBatch(ExperiencePoolBatch batch, int size){
		ExperiencePoolBatch b = pool.getBatch(batch, sampling.next(size));
		
		// potentially replace some batch items by samples from the priority buffer
		if(config.prioritySamplingFactor > 0){
			for(int i=0;i<size;i++){
				if(priorityBuffer.size() > 0){
					if(random.nextDouble() < config.prioritySamplingFactor){
						// how to best sample from priority buffer?
						float highest = priorityBuffer.lastKey();
						float lowest = priorityBuffer.firstKey();
						float key = lowest + random.nextFloat()*(highest-lowest);
						
						ExperiencePoolSample s = priorityBuffer.ceilingEntry(key).getValue();
						
						ExperiencePoolSample replace = b.getSample(i);
						s.copyInto(replace);
					}
				}
			}
		}
		
		return b;
	}
	
	public void addBatch(Tensor priority, ExperiencePoolBatch batch){
		if(config.prioritySamplingFactor > 0 ){
			for(int i=0;i<priority.size();i++){
				float p = priority.get(i);
				if(priorityBuffer.size() < config.prioritySamplingSize
						|| priorityBuffer.lowerKey(p) != null){
					
					ExperiencePoolSample save = new ExperiencePoolSample();
					batch.getSample(i).copyInto(save);
					priorityBuffer.put(p, save);
				}
			}
			
			while(priorityBuffer.size() > config.prioritySamplingSize){
				priorityBuffer.pollFirstEntry();
			}
		}
	}
}
