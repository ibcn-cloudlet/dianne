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
package be.iminds.iot.dianne.dataset.adapters;

import java.util.Map;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;

public abstract class AbstractDatasetAdapter implements Dataset {

	protected Dataset data;
	protected String name;
	
	protected boolean targetDimsSameAsInput = false;

	private Sample temp;
	
	@Reference
	void setDataset(Dataset d){
		this.data = d;
	}
	
	@Activate
	void activate(Map<String, Object> properties) {
		this.name = (String)properties.get("name");

		// mark if targetDims are same as inputs
		// often requires adapters to also adapt target
		int[] inputDims = data.inputDims();
		if(inputDims != null){
			int[] targetDims = data.targetDims();
			if(inputDims.length == targetDims.length){
				targetDimsSameAsInput = true;
				for(int i=0;i<inputDims.length;i++){
					if(inputDims[i] != targetDims[i]){
						targetDimsSameAsInput = false;
						break;
					}
				}
			}
		}
		
		configure(properties);
	}
	
	protected abstract void configure(Map<String, Object> properties);
	
	@Override
	public String getName(){
		return name;
	}
	
	@Override
	public int[] inputDims(){
		return data.inputDims();
	}
	
	@Override
	public int[] targetDims(){
		return data.targetDims();
	}
	
	@Override
	public int size() {
		return data.size();
	}

	@Override
	public synchronized Sample getSample(Sample s, int index) {
		temp = data.getSample(temp, index);
		if(s == null){
			s = new Sample();
		}
		adaptSample(temp, s);
		return s;
	};
	
	protected abstract void adaptSample(Sample original, Sample adapted);
	
	@Override
	public String[] getLabels(){
		return data.getLabels();
	}
}
