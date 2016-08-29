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

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.ConfigurationPolicy;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This Dataset adapter allows you to wrap a Dataset in a new Dataset with only
 * a subset of the labels available. The other labels can be either ignored or 
 * be aggregated into one "other" class.
 * 
 * Example config:
   {
	"dataset":"MNIST",
	"adapter":"be.iminds.iot.dianne.dataset.adapters.LabelAdapter",
	"name": "MNIST 1vs all",
	"other": true,
	"labels":["1"]
	}
 * 
 * @author tverbele, ejodconi
 *
 */
@Component(
	service={Dataset.class},
	configurationPolicy=ConfigurationPolicy.REQUIRE,
	configurationPid="be.iminds.iot.dianne.dataset.adapters.LabelAdapter")
public class DatasetLabelAdapter extends AbstractDatasetAdapter {
	
	// label indices
	private int[] labelIndices;
	private int[] targetDims;
	private String[] labels;
	// add "other" category
	private boolean other;
	
	@Override
	public int[] targetDims(){
		return targetDims;
	}
	
	protected void configure(Map<String, Object> config){
		// whether or not to aggregate unspecified labels into "other" class
		this.other = Boolean.parseBoolean((String)config.get("other"));
		String[] labels = (String[])config.get("labels");
		this.labels = new String[labels.length+ (other ? 1 : 0)];
		System.arraycopy(labels, 0, this.labels, 0, labels.length);
		if(other){
			this.labels[labels.length] = "other";
		}
		this.labelIndices = new int[labels.length];
		List<String> labelsList = Arrays.asList(data.getLabels());
		for(int i=0;i<labels.length;i++){
			labelIndices[i] = labelsList.indexOf(labels[i]);
		}
		this.targetDims = new int[]{this.labels.length};
	}
	
	@Override
	protected void adaptSample(Sample original, Sample adapted) {
		adapted.input = original.input.copyInto(adapted.input);
		if(adapted.target == null){
			adapted.target = new Tensor(labels.length);
		}
		adapted.target.fill(0.0f);
		for(int i=0;i<labelIndices.length;i++){
			adapted.target.set(original.target.get(labelIndices[i]), i);
		}
		if(other){
			if(TensorOps.sum(adapted.target)==0){
				adapted.target.set(1.0f, labels.length-1);
			}
		}
	}
	
	@Override
	public String[] getLabels(){
		return this.labels;
	}
	
}
