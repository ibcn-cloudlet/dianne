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
	"range": [0,60000]
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
	private int start = 0;
	private int end = 1;
	
	private int[] indices = null;
	
	protected void configure(Map<String, Object> properties) {
		Object r = properties.get("range");
		if(r instanceof String[]){
			String[] range = (String[]) r;
			if(range.length == 1){
				end = Integer.parseInt(range[0]);
			} else if(range.length == 2){
				start = Integer.parseInt(range[0]);
				end = Integer.parseInt(range[1]);
			} else {
				indices = new int[range.length];
				for(int i=0;i<range.length;i++){
					indices[i] = Integer.parseInt(range[i]);
				}
			}
		} else {
			end = Integer.parseInt((String)r);
		}
	}
	
	@Override
	public int size() {
		if(indices != null)
			return indices.length;
		
		return end-start;
	}

	@Override
	public Sample getSample(Sample s, int index){
		Sample result = null;
		if(indices != null){
			result = data.getSample(s, indices[index]);
		} else {
			result = data.getSample(s, start+index);
		}
		// in case dataset is remote, we need to make sure we copy here...
		if(s == null){
			s = new Sample();
		}
		s.input = result.input.copyInto(s.input);
		s.target = result.target.copyInto(s.target);
		return s;
	}

	@Override
	protected void adaptSample(Sample original, Sample adapted) {}

}
