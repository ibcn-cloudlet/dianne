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

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This Dataset adapter allows you to wrap a Dataset in a new Dataset with framed
 * versions of the input samples. 
 * 
 * Example config:
   {
	"dataset":"ImageNet",
	"adapter":"be.iminds.iot.dianne.dataset.adapters.FrameAdapter",
	"name": "ImageNet 231x231",
	"frame": [3,231,231]
   }
 * 
 * @author tverbele
 *
 */
@Component(
	service={Dataset.class},
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.FrameAdapter")
public class DatasetFrameAdapter extends AbstractDatasetAdapter {

	private int[] dims;
	
	protected void configure(Map<String, Object> properties) {
		String[] d = (String[])properties.get("frame");
		dims = new int[d.length];
		for(int i=0;i<d.length;i++){
			dims[i] = Integer.parseInt(d[i]);
		}
	}
	
	public int[] inputDims(){
		return dims;
	}
	
	public int[] targetDims(){
		if(targetDimsSameAsInput){
			return dims;
		}
		return data.targetDims();
	}
	
	@Override
	protected void adaptSample(Sample original, Sample adapted) {
		adapted.input = TensorOps.frame(adapted.input, original.input, dims);
		if(targetDimsSameAsInput){
			adapted.target = TensorOps.frame(adapted.target, original.target, dims);
		} else {
			adapted.target = original.target.copyInto(adapted.target);
		}
	}

}
