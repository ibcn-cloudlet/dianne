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
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This Dataset adapter will set the target output the same as the input.
 * 
 * Can be used to train AutoEncoders on an existing (classification) Dataset
 * 
 * @author tverbele
 *
 */
@Component(
	service={Dataset.class},	
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.AutoencoderAdapter")
public class DatasetAutoencoderAdapter extends AbstractDatasetAdapter {
	
	// Noise
	private float sigma = 0.0f;
	private Tensor noise = null;

	@Override
	protected void configure(Map<String, Object> properties) {
		if(properties.containsKey("encoder_noise"))
			this.sigma = Float.parseFloat((String) properties.get("encoder_noise"));
	}
	
	@Override
	public String[] getLabels(){
		return null;
	}

	@Override
	public int[] targetDims() {
		return inputDims();
	};

	@Override
	protected void adaptSample(Sample original, Sample adapted) {
		adapted.input = original.input.copyInto(adapted.input);
		adapted.target = original.input.copyInto(adapted.target);
		
		if(sigma > 0) {
			if(noise == null || !adapted.input.hasDim(noise.dims())) {
				noise = new Tensor(noise.dims());
			}
			
			noise.randn();
			TensorOps.add(adapted.input, adapted.input, sigma, noise);
		}
	}

}
