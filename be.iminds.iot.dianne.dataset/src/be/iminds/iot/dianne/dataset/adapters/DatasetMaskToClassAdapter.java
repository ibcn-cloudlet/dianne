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

import be.iminds.iot.dianne.api.dataset.AbstractDatasetAdapter;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This Dataset adapter will convert a segmentation mask to a class (true or false)
 * 
 * If no white pixels are in the mask, the class is false, true otherwise.
 * 
 * @author tverbele
 *
 */
@Component(
	service={Dataset.class},	
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.MaskToClassAdapter")
public class DatasetMaskToClassAdapter extends AbstractDatasetAdapter {

	private String[] labels = new String[]{"True","False"};
	private int[] targetDims = new int[]{2};
	private boolean center = false;
	private float threshold = 0.5f;
	
	@Override
	protected void configure(Map<String, Object> properties) {
		if(properties.containsKey("center")){
			center = Boolean.parseBoolean((String)properties.get("center"));
		}
		
		if(properties.containsKey("threshold")){
			threshold = Float.parseFloat((String)properties.get("threshold"));
		}
	}
	
	@Override
	public String[] getLabels(){
		return labels;
	}

	@Override
	public int[] targetDims() {
		return targetDims;
	};

	@Override
	protected void adaptSample(Sample original, Sample adapted) {
		adapted.input = original.input.copyInto(adapted.input);
		if(adapted.target == null){
			adapted.target = new Tensor(2);
		} else {
			adapted.target.reshape(2);
		}
		
		boolean clazz = false;
		if(center){
			int[] dims = original.target.dims();
			int centerW = dims[dims.length-1]/2;
			int centerH = dims[dims.length-2]/2;
			if(original.target.get(0, centerH, centerW) > threshold){
				clazz = true;
			}
		} else {
			float sum = TensorOps.sum(original.target);
			if(sum > threshold){
				clazz = true;
			} 
		}
		
		if(clazz){
			// true
			adapted.target.set(1.0f, 0);
			adapted.target.set(0.0f, 1);
		} else {
			// false
			adapted.target.set(0.0f, 0);
			adapted.target.set(1.0f, 1);
		}
	}

}
