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
 * a (batched) input and output Tensor
 * 
 * In a batch, the input/output from Sample represents all batches in a single 
 * Tensor with dim 0 = batchSize. Separate Tensors per sample are available in 
 * the inputSamples/outputSamples arrays. 
 * 
 * In case the Dataset has variable input dims, a single batched Tensor is 
 * impossible to construct, in which case input/output are null and the inputSamples
 * outputSamples arrays contain separate Tensor objects for each sample in the batch.
 * 
 * @author tverbele
 *
 */
public class Batch extends Sample {
	
	public final int batchSize;
	public final Tensor[] inputSamples;
	public final Tensor[] outputSamples;
	
	
	public Batch(Tensor i, Tensor o){
		super(i, o);
		// TODO check i.size(0)==o.size(0)??
		batchSize = i.size(0);
		inputSamples = new Tensor[batchSize];
		outputSamples = new Tensor[batchSize];
		for(int k=0;k<batchSize;k++){
			inputSamples[k] = i.select(0, k);
			outputSamples[k] = o.select(0, k);
		}
	}

	public Batch(Tensor[] i, Tensor[] o){
		super(null, null); 
		// no single batch Tensor exists
		// only an array of separate tensors
		// TODO check i.length == o.length??
		batchSize = i.length;
		inputSamples = i;
		outputSamples = o;
	}
	
	public int getSize(){
		return batchSize;
	}
	
	public Tensor getInput(int i){
		return inputSamples[i];
	}
	
	public Tensor getOutput(int i){
		return outputSamples[i];
	}
}
