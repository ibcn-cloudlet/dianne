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
 * A helper class for representing a batch of a dataset, a combination of
 * a (batched) input and target Tensor
 * 
 * In a batch, the input/target from Sample represents all batches in a single 
 * Tensor with dim 0 = batchSize. Separate Tensors per sample are available in 
 * the inputSamples/targetSamples arrays. 
 * 
 * In case the Dataset has variable input dims, a single batched Tensor is 
 * impossible to construct, in which case input/target are null and the inputSamples
 * targetSamples arrays contain separate Tensor objects for each sample in the batch.
 * 
 * @author tverbele
 *
 */
public class Batch extends Sample {
	
	public Sample[] samples;
	
	protected Batch(){};
	
	public Batch(int batchSize, int[] inputDims, int[] targetDims){
		super(new Tensor(batchSize, inputDims),new Tensor(batchSize, targetDims));
		this.samples = new Sample[batchSize];
		for(int i = 0; i< batchSize; i++){
			this.samples[i] = new Sample(input.select(0, i), target.select(0, i));
		}
	}
	
	public Batch(Tensor input, Tensor target){
		super(input, target);
		int batchSize = input.size(0);
		this.samples = new Sample[batchSize];
		for(int i = 0; i< batchSize; i++){
			this.samples[i] = new Sample(input.select(0, i), target.select(0, i));
		}
	}

	public Batch(int batchSize){
		// fill with empty samples...
		this.samples = new Sample[batchSize];
		for(int i = 0; i< batchSize; i++){
			this.samples[i] = new Sample();
		}
	}
	
	public Batch(Sample[] samples){
		// no single batch Tensor exists
		// only an array of separate tensors
		this.samples = samples;
	}
	
	public int getSize(){
		return samples.length;
	}
	
	public Sample getSample(int i){
		return samples[i];
	}
	
	public Tensor getInput(int i){
		return samples[i].input;
	}
	
	public Tensor getTarget(int i){
		return samples[i].target;
	}
}
