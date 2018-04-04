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
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This Dataset adapter extends the Dataset by also rotating samples
 * 
 * Configure the adapter by providing a theta angle range 
 * rotationTheta = [min, max]
 * 
 * Optionally one can also set the center of rotation using either
 * 
 * rotationCenter = [x, y]  // set x,y as center of rotation
 * rotationCenter = true  // center around image center
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

	private float minTheta = (float)-Math.PI;
	private float maxTheta = (float)Math.PI;
	
	private int[] center = null;
	private boolean middle = false;
	private boolean zeropad = false;
	
	private Random r = new Random(System.currentTimeMillis());
	
	protected void configure(Map<String, Object> properties) {
		if(properties.containsKey("rotationTheta")){
			Object t = properties.get("rotationTheta");
			if(t instanceof String[]){
				minTheta = Float.parseFloat(((String[]) t)[0].trim());
				maxTheta = Float.parseFloat(((String[]) t)[1].trim());
			} else {
				maxTheta = Float.parseFloat((String) t);
				minTheta = -maxTheta;
			}
		}
		
		if(properties.containsKey("rotationCenter")){
			Object c = properties.get("rotationCenter");
			// center = [x,y] means : center rotation around x,y
			if(c instanceof String[]){
				center[0] = Integer.parseInt(((String[])c)[0].trim());
				center[1] = Integer.parseInt(((String[])c)[1].trim());
			} else {
				// center = true means: center around image center	
				middle = Boolean.parseBoolean((String)c);
			}
		}
		
		if(properties.containsKey("zeropad")){
			zeropad = Boolean.parseBoolean(properties.get("zeropad").toString());
		}
	}
	
	@Override
	protected void adaptSample(Sample original, Sample adapted) {
		float theta = minTheta+r.nextFloat()*(maxTheta-minTheta);
		
		int[] dims = original.input.dims();
		int height = dims.length == 3 ? dims[1] : dims[0];
		int width = dims.length == 3 ? dims[2] : dims[1];
		
		int center_x = r.nextInt(width);
		int center_y = r.nextInt(height);
		if(middle){
			center_x = width/2;
			center_y = height/2;
		} else if(center != null){
			center_x = center[0];
			center_y = center[1];
		}
		
		adapted.input = TensorOps.rotate(adapted.input, original.input, theta, center_x, center_y, zeropad);
		if(targetDimsSameAsInput){
			adapted.target = TensorOps.rotate(adapted.target, original.target, theta, center_x, center_y, zeropad);
		} else {
			adapted.target = original.target.copyInto(adapted.target);
		}
	}

	// TODO implement this as (native) TensorOp
	
}
