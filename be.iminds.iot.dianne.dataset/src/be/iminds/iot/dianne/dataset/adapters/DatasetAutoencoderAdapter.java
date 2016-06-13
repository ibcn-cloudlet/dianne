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
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This Dataset adapter will set the target output the same as the input.
 * 
 * Can be used to train AutoEncoders on an existing (classification) Dataset
 * 
 * @author tverbele
 *
 */
@Component(configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.AutoencoderAdapter")
public class DatasetAutoencoderAdapter implements Dataset {

	private Dataset data;
	private String name;

	@Reference
	void setDataset(Dataset d){
		this.data = d;
	}
	
	@Activate
	void activate(Map<String, Object> properties) {
		this.name = (String)properties.get("name");
	}
	
	@Override
	public String getName(){
		return name;
	}
	
	@Override
	public int[] inputDims(){
		return data.inputDims();
	}
	
	@Override
	public int[] outputDims(){
		return data.inputDims();
	}
	
	@Override
	public int size() {
		return data.size();
	}

	@Override
	public Tensor getInputSample(Tensor t, int index) {
		return data.getInputSample(t, index);
	}

	@Override
	public Tensor getOutputSample(Tensor t, int index) {
		return data.getInputSample(t, index);
	}
	
	@Override
	public String[] getLabels(){
		return null;
	}

}
