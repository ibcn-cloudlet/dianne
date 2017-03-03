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

/**
 * A Dataset is a collection of input data and the corresponding targets.
 * 
 * The input tensor can be an n-dimensional input (likely 1-D (sequence) or 2-D (images))
 * The input values range between 0 and 1.
 * 
 * For classification problems the target tensor can be a 1 dimensional vector filled with 
 * zeros and only 1 for the classes that are represented by the corresponding input sample.
 * 
 * For regression problems the target tensor can be arbitrary.
 * 
 * Each item from the target tensor corresponds with a human-readable label.
 * 
 * A Dataset can be used for supervised training of a neural network.
 * 
 * @author tverbele
 *
 */
public interface Dataset {
	
	/**
	 * Returns the number of samples in the dataset
	 * 
	 * @return the number of samples in the dataset
	 */
	int size();

	int[] inputDims();
	
	String inputType();
	
	int[] targetDims();
	
	String targetType();
	
	/**
	 * Returns a sample from the dataset
	 * @param index the index to fetch, should be smaller than size()
	 * @return the sample at position index
	 */
	default Sample getSample(final int index){
		return getSample(null, index);
	}
	
	/**
	 * Fetches a sample from the dataset and puts the data into the provided Sample object
	 * If the provided Sample object is null a new one will be created and returned.
	 * @param s the Sample object to put the data into
	 * @param index the index to fetch, should be smaller than size()
	 * @return the sample at position index
	 */
	Sample getSample(Sample s, final int index);
	
	/**
	 * Get raw sample data
	 * 
	 * Override this method to avoid Tensor creation in the Dataset implementation
	 * 
	 * @param index the index to fetch, should be smaller than size()
	 * @return the raw sample data at position index
	 */
	default RawSample getRawSample(final int index){
		Sample s = getSample(index);
		return new RawSample(s.input.dims(), s.input.get(), s.target.dims(), s.target.get());
	}
	
	/**
	 * Returns a batch from the dataset with samples at indices.
	 * @param indices the indices of the samples to fetch
	 * @return batch with samples at indices
	 */
	default Batch getBatch(final int...indices) {
		return getBatch(null, indices);
	}
	
	/**
	 * Fetches the samples at indices from the dataset and puts the data into the provided Batch object
	 * If the provided Batch object is null a new one will be created and returned.
	 * 
	 * In case the dataset has varying input dims, a single batch cannot be constructed an
	 * an InstantiationError will be thrown
	 * 
	 * @param b Batch object to put the data in
	 * @param indices indices to fetch
	 * @return
	 */
	default Batch getBatch(Batch b, final int...indices){
		if(b == null){
			int[] inputDims = inputDims();
			int[] targetDims = targetDims();

			if(inputDims == null){
				throw new InstantiationError("Cannot create a batch when dataset has no fixed input dimensions");
			} else {
				b = new Batch(indices.length, inputDims, targetDims);
			}
		}
		
		for(int i=0;i<indices.length;i++){
			getSample(b.samples[i], indices[i]);
		}
		
		return b;
	}
	
	/**
	 * Fetches raw batched data from the dataset
	 * 
	 * Override this method to avoid Tensor creation in the Dataset implementation
	 *
	 * @param indices indices to fetch
	 * @return
	 */
	default RawBatch getRawBatch(final int...indices){
		Batch b = getBatch(indices);
		return new RawBatch(b.input.dims(), b.input.get(), b.target.dims(), b.target.get());
	}
	
	/**
	 * A human-readable name for this dataset
	 * 
	 * @return dataset name
	 */
	String getName();
	
	/**
	 * Get human-readable names for the classes represented in an target vector
	 * 
	 * @return human-readable dataset labels
	 */
	String[] getLabels();
}
