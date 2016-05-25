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
@Component(configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.FrameAdapter",
	property={"aiolos.unique=true"})
public class DatasetFrameAdapter implements Dataset {

	private Dataset data;
	private String name;

	private int[] dims;
	
	@Reference
	void setDataset(Dataset d){
		this.data = d;
	}
	
	@Activate
	void activate(Map<String, Object> properties) {
		this.name = (String)properties.get("name");
		String[] d = (String[])properties.get("frame");
		dims = new int[d.length];
		for(int i=0;i<d.length;i++){
			dims[i] = Integer.parseInt(d[i]);
		}
	}
	
	@Override
	public String getName(){
		return name;
	}
	
	@Override
	public int[] inputDims(){
		return dims;
	}
	
	@Override
	public int[] outputDims(){
		return data.outputDims();
	}
	
	@Override
	public int size() {
		return data.size();
	}

	@Override
	public Tensor getInputSample(Tensor t, int index) {
		return TensorOps.frame(t, data.getInputSample(index), dims);
	}

	@Override
	public Tensor getOutputSample(Tensor t, int index) {
		return data.getOutputSample(t, index);
	}
	
	@Override
	public String[] getLabels(){
		return data.getLabels();
	}

}
