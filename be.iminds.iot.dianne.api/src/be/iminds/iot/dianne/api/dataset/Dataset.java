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

	int[] inputDims();
	
	int[] outputDims();
	
	/**
	 * Returns a sample from the dataset
	 * @param index the index to fetch, should be smaller than size()
	 * @return the sample at position index
	 */
	default Sample getSample(final int index){
		return getSample(null, index);
	}
	
	default Sample getSample(Sample s, final int index){
		if(s == null)
			return new Sample(getInputSample(index), getOutputSample(index));
		
		getInputSample(s.input, index);
		getOutputSample(s.output, index);
		return s;
	}
	
	default Batch getBatch(final int...indices) {
		return getBatch(null, indices);
	}
	
	default Batch getBatch(Batch b, final int...indices){
		if(b == null){
			int[] inputDims = inputDims();
			int[] outputDims = outputDims();

			if(inputDims == null){
				Tensor[] inputs = new Tensor[indices.length];
				for(int i=0;i<indices.length;i++){
					inputs[i] = new Tensor(inputDims);
				}
				
				Tensor[] outputs = new Tensor[indices.length];
				for(int i=0;i<indices.length;i++){
					outputs[i] = new Tensor(outputDims);
				}
				
				b = new Batch(inputs, outputs);
			} else {
				int[] batchedInputDims = new int[inputDims.length+1];
				batchedInputDims[0] = indices.length;
				for(int i=0;i<inputDims.length;i++){
					batchedInputDims[i+1] = inputDims[i];
				}
				
				int[] batchedOutputDims = new int[outputDims.length+1];
				batchedOutputDims[0] = indices.length;
				for(int i=0;i<outputDims.length;i++){
					batchedOutputDims[i+1] = outputDims[i];
				}
				
				b = new Batch(new Tensor(batchedInputDims), new Tensor(batchedOutputDims));
			}
		}
		
		for(int i=0;i<indices.length;i++){
			getInputSample(b.inputSamples[i], indices[i]);
			getOutputSample(b.outputSamples[i], indices[i]);
		}
		
		return b;
	}
	
	/**
	 * Get an input sample from the dataset
	 * 
	 * @param index the index to fetch, should be smaller than size()
	 * @return the input sample at position index
	 */
	default Tensor getInputSample(final int index){
		return getInputSample(null, index);
	}
		
	Tensor getInputSample(Tensor t, final int index);
	
	/**
	 * Get an output vector from the dataset
	 * 
	 * @param index the index to fetch, should be smaller than size()
	 * @return the output vector corresponding with input sample index
	 */
	default Tensor getOutputSample(final int index){
		return getOutputSample(null, index);
	}
	
	Tensor getOutputSample(Tensor t, final int index);
	
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
