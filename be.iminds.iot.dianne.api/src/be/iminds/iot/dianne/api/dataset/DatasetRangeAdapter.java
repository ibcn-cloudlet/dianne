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

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This Dataset adapter allows you to wrap a Dataset in a new Dataset with only
 * a subset of the samples. Can for example be used to split up a Dataset into
 * a training, validation and test set.
 * 
 * @author tverbele
 *
 */
public class DatasetRangeAdapter implements Dataset {

	private Dataset data;
	
	// start and end index
	private int start;
	private int end;
	
	/**
	 * Creates a new DatasetRangeAdapter
	 * 
	 * @param data the dataset to wrap
	 * @param start the start index for the new dataset
	 * @param end the end index of the new dataset
	 */
	public DatasetRangeAdapter(Dataset data, int start, int end) {
		this.start = start;
		this.end = end;
		this.data = data;
	}
	
	@Override
	public String getName(){
		return data.getName();
	}
	
	@Override
	public int size() {
		return end-start;
	}

	@Override
	public Tensor getInputSample(int index) {
		return data.getInputSample(start+index);
	}

	@Override
	public Tensor getOutputSample(int index) {
		return data.getOutputSample(start+index);
	}
	
	@Override
	public String[] getLabels(){
		return data.getLabels();
	}

}
