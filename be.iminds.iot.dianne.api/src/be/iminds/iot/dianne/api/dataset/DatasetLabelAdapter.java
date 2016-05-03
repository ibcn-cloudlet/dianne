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
package be.iminds.iot.dianne.api.dataset;

import java.util.Arrays;
import java.util.List;

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This Dataset adapter allows you to wrap a Dataset in a new Dataset with only
 * a subset of the labels available. The other labels can be either ignored or 
 * be aggregated into one "other" class.
 * 
 * @author tverbele
 *
 */
public class DatasetLabelAdapter implements Dataset {
	
	private Dataset data;
	
	// label indices
	private int[] labelIndices;
	private String[] labels;
	// add "other" category
	private boolean other;
	
	/**
	 * Create a new DatasetLabelAdapter
	 * 
	 * @param f tensor factory to use for creating Tensors
	 * @param data the dataset to wrap
	 * @param labels the labels to keep
	 * @param other whether or not the other labels should be aggregated into an "other" class
	 * 
	 */
	public DatasetLabelAdapter(Dataset data, String[] labels, boolean other) {
		this.data = data;
		this.other = other;
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
	}
	
	@Override
	public String getName(){
		return data.getName();
	}
	
	@Override
	public int size() {
		return data.size();
	}
	
	@Override
	public Tensor getInputSample(int index) {
		return data.getInputSample(index);
	}

	@Override
	public Tensor getOutputSample(int index) {
		// TODO adapt outputsample
		Tensor t = data.getOutputSample(index);
		Tensor t2 = new Tensor(labels.length);
		for(int i=0;i<labelIndices.length;i++){
			t2.set(t.get(labelIndices[i]), i);
		}
		if(other){
			if(TensorOps.sum(t2)==0){
				t2.set(1.0f, labels.length-1);
			}
		}
		return t2;
	}
	
	@Override
	public String[] getLabels(){
		return labels;
	}

}
