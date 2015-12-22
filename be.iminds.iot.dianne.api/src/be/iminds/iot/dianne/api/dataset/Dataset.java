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
 * A Dataset is a collection of input data and the corresponding output classes.
 * 
 * The input tensor can be an n-dimensional input (likely 1-D (sequence) or 2-D (images))
 * The input values range between 0 and 1.
 * 
 * The output tensor is a 1 dimensional vector filled with zeros and only 1 for the classes
 * that are represented by the corresponding input sample. 
 * 
 * Each item from the output tensor corresponds with a human-readable label.
 * 
 * A Dataset can be used for supervised training of a neural network.
 * 
 * @author tverbele
 *
 */
public interface Dataset {
	
	/**
	 * Returns the number of input-output samples in the dataset
	 * 
	 * @return the number of input-output samples in the dataset
	 */
	int size();

	/**
	 * Get an input sample from the dataset
	 * 
	 * @param index the index to fetch, should be smaller than size()
	 * @return the input sample at position index
	 */
	Tensor getInputSample(final int index);
		
	/**
	 * Get an output vector from the dataset
	 * 
	 * @param index the index to fetch, should be smaller than size()
	 * @return the output vector corresponding with input sample index
	 */
	Tensor getOutputSample(final int index);
	
	/**
	 * A human-readable name for this dataset
	 * 
	 * @return dataset name
	 */
	String getName();
	
	/**
	 * Get human-readable names for the classes represented in an output vector
	 * 
	 * @return human-readable dataset labels
	 */
	String[] getLabels();
}
