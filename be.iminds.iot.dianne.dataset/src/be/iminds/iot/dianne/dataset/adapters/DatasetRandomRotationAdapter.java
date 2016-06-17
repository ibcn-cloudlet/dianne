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

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This Dataset adapter extends the Dataset by also rotating samples
 * 
 * Configure the adapter by providing a theta angle range 
 * theta = [min, max]
 * 
 * Optionally one can also set the center of rotation using either
 * 
 * center = [x, y]  // set x,y as center of rotation
 * center = true  // center around image center
 * 
 * If not configured a random center will be generated for each getSample()
 * 
 * @author tverbele
 *
 */
@Component(
	service={Dataset.class},
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.RandomRotationAdapter")
public class DatasetRandomRotationAdapter extends AbstractDatasetAdapter {

	private float minTheta = 0;
	private float maxTheta = 0;
	
	private int[] center = null;
	private boolean middle = false;
	
	private Random r = new Random(System.currentTimeMillis());
	
	protected void configure(Map<String, Object> properties) {
		Object t = properties.get("theta");
		if(t instanceof String[]){
			minTheta = Integer.parseInt(((String[]) t)[0]);
			maxTheta = Integer.parseInt(((String[]) t)[1]);
		} else {
			minTheta = Integer.parseInt((String) t);
			maxTheta = minTheta;
		}
		if(properties.containsKey("center")){
			Object c = properties.get("center");
			// center = [x,y] means : center rotation around x,y
			if(c instanceof String[]){
				center[0] = Integer.parseInt(((String[])c)[0]);
				center[1] = Integer.parseInt(((String[])c)[1]);
			} else {
				// center = true means: center around image center	
				middle = Boolean.parseBoolean((String)c);
			}
		}
	}
	
	@Override
	protected void adaptSample(Sample original, Sample adapted) {
		float theta = minTheta+r.nextFloat()*(maxTheta-minTheta);
		
		adapted.input = rotate(adapted.input, original.input, theta);
		if(targetDimsSameAsInput){
			adapted.target = rotate(adapted.target, original.target, theta);
		} else {
			adapted.target = original.target.copyInto(adapted.target);
		}
	}

	// TODO implement this as (native) TensorOp
	private Tensor rotate(Tensor res, final Tensor t, float theta){
		float[] rotatedData = new float[t.size()];
		
		int[] dims = t.dims();
		float[] data = t.get();
		
		int channels = dims.length == 3 ? dims[0] : 1;
		int height = dims.length == 3 ? dims[1] : dims[0];
		int width = dims.length == 3 ? dims[2] : dims[1];
		
		double sin_theta = Math.sin(theta);
		double cos_theta = Math.cos(theta);
		
		
		
		int center_x = r.nextInt(width);
		int center_y = r.nextInt(height);
		if(middle){
			center_x = width/2;
			center_y = height/2;
		} else if(center != null){
			center_x = center[0];
			center_y = center[1];
		}
		
		for(int c = 0; c < channels ; c++){
			for(int j=0;j<height;j++){
				for(int i=0;i<width;i++){
					
					int heightIndex = (int)((i - center_x)*sin_theta + (j - center_y)*cos_theta + center_y);
					int widthIndex = (int)((i - center_x)*cos_theta - (j - center_y)*sin_theta + center_x);
					
					if(heightIndex < 0 || heightIndex >= height
							|| widthIndex < 0 || widthIndex >= width){
						rotatedData[c*width*height+j*width+i] = 0;
					} else {
						rotatedData[c*width*height+j*width+i] = data[c*width*height+heightIndex*width+widthIndex]; 
					}
				}
			}
		}
		
		if(res == null){
			res = new Tensor(rotatedData, dims);
		} else {
			res.set(rotatedData);
		}
		return res;
	}
}
