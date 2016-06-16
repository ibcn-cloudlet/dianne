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

/**
 * This Dataset adapter allows you to wrap a Dataset in a new Dataset with only
 * a subset of the samples. Can for example be used to split up a Dataset into
 * a training, validation and test set.
 * 
 * Example config:
   {
	"dataset":"MNIST",
	"adapter":"be.iminds.iot.dianne.dataset.adapters.RangeAdapter",
	"name": "MNIST train set",
	"start": 0,
	"end": 60000 
   }
 * 
 * @author tverbele
 *
 */
@Component(
	service={Dataset.class},
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.RangeAdapter")
public class DatasetRangeAdapter extends AbstractDatasetAdapter {

	// start and end index
	private int start;
	private int end;
	
	protected void configure(Map<String, Object> properties) {
		this.start = Integer.parseInt((String)properties.get("start"));
		this.end = Integer.parseInt((String)properties.get("end"));
	}
	
	@Override
	public int size() {
		return end-start;
	}

	@Override
	public Sample getSample(Sample s, int index){
		return data.getSample(s, start+index);
	}

	@Override
	protected void adaptSample(Sample original, Sample adapted) {}

}
