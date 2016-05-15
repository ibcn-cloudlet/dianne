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
package be.iminds.iot.dianne.api.rnn.dataset;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A SequenceDataset is a Dataset consisting out of a long sequence of entries.
 * 
 * This dataset is used to train recurrent neural networks, where the goal is to
 * predict the next entry in the sequence from the history.
 * 
 * @author tverbele
 *
 */
public interface SequenceDataset extends Dataset {
	
	/**
	 * Returns the number of entries in the dataset
	 * 
	 * @return the number of entries in the dataset
	 */
	int size();

	/**
	 * Return a subset of the sequence, i.e. when training on sequences with limited length
	 * @param index start index of the first input
	 * @param length number of inputs requested
	 * @return a sequence of length+1 items: length inputs + 1 corresponding output
	 */
	Tensor[] getSequence(final int index, final int length);
	
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
