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
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn.sampling;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;

/**
 * Makes sure you sample equally from each class 
 * 
 * Not very performant, needs to index the whole dataset on initialization...
 * 
 * @author tverbele
 *
 */
public class UniformClassSamplingStrategy implements SamplingStrategy{

	private Random random = new Random(System.currentTimeMillis());
	
	private final Dataset dataset;
	
	private final List<List<Integer>> indexMap;
	
	private int clazz = 0;
	
	public UniformClassSamplingStrategy(Dataset dataset) {
		this.dataset = dataset;
		
		int noClasses = dataset.targetDims()[0] + 1;
		
		this.indexMap = new ArrayList<List<Integer>>(noClasses);
		for(int i = 0 ; i< noClasses ; i++){
			indexMap.add(new ArrayList<Integer>());
		}
		
		Sample s = null;
		for(int i = 0; i< dataset.size(); i++){
			s = dataset.getSample(s, i);
			float[] data = s.target.get();
			boolean other = true;
			for(int k=0;k<data.length;k++){
				if(data[k] > 0.5f){
					indexMap.get(k).add(i);
					other = false;
				}
			}
			if(other)
				indexMap.get(data.length).add(i);
		}
	}
	
	@Override
	public int next() {
		List<Integer> list = indexMap.get(clazz++);
		if(clazz >= indexMap.size()){
			clazz = 0;
		}
		int listIndex = random.nextInt(list.size());
		return list.get(listIndex);
	}

	@Override
	public int[] next(int count){
		int[] indices = new int[count];
		for(int i=0;i<count;i++){
			indices[i] = next();
		}
		return indices;
	}
}
