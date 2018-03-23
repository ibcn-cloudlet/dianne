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
import java.util.Random;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.AbstractDatasetAdapter;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This Dataset adapter extends the Dataset by also flipping samples either vertical or horizontal
 * 
 * Configure by providing a vflip and/or hflip probability that a sample is flipped.
 * 
 * @author tverbele
 *
 */
@Component(
	service={Dataset.class},
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.RandomFlipAdapter")
public class DatasetRandomFlipAdapter extends AbstractDatasetAdapter {

	private float vflip = 0;
	private float hflip = 0;
	
	private Random r = new Random(System.currentTimeMillis());
	
	protected void configure(Map<String, Object> properties) {
		if(properties.containsKey("vflip")){
			vflip = Float.parseFloat((String)properties.get("vflip"));
		}
		if(properties.containsKey("hflip")){
			hflip = Float.parseFloat((String)properties.get("hflip"));
		}
	}
	
	@Override
	protected void adaptSample(Sample original, Sample adapted) {
		boolean vertical = r.nextFloat() <= vflip;
		boolean horizontal = r.nextFloat() <= hflip;
		
		adapted.input = flip(adapted.input, original.input, vertical, horizontal);
		if(targetDimsSameAsInput){
			adapted.target = flip(adapted.target, original.target, vertical, horizontal);
		} else {
			adapted.target = original.target.copyInto(adapted.target);
		}
	}

	// TODO implement this as (native) TensorOp
	private Tensor flip(Tensor res, final Tensor t, boolean vertical, boolean horizontal){
		float[] flippedData = new float[t.size()];
		
		int[] dims = t.dims();
		float[] data = t.get();
		
		int channels = dims.length == 3 ? dims[0] : 1;
		int height = dims.length == 3 ? dims[1] : dims[0];
		int width = dims.length == 3 ? dims[2] : dims[1];
		
		for(int c = 0; c < channels ; c++){
			for(int j=0;j<height;j++){
				for(int i=0;i<width;i++){
					int heightIndex = vertical ? height-j-1 : j;
					int widthIndex = horizontal ? width-i-1 : i;
					
					flippedData[c*width*height+j*width+i] = data[c*width*height+heightIndex*width+widthIndex]; 
				}
			}
		}
		
		if(res == null){
			res = new Tensor(flippedData, dims);
		} else {
			res.set(flippedData);
		}
		return res;
	}
}
