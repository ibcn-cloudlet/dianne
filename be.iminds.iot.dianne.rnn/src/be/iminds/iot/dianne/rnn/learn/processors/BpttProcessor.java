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
package be.iminds.iot.dianne.rnn.learn.processors;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rnn.dataset.SequenceDataset;
import be.iminds.iot.dianne.nn.learn.processors.AbstractProcessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class BpttProcessor extends AbstractProcessor {

	// dataset
	protected final SequenceDataset dataset;
	// sample strategy
	protected final SamplingStrategy sampling;
	// error criterion
	protected final Criterion criterion;
	// sequence length
	protected final int sequenceLength;
	// backprop all items in sequence or only the last one
	protected final boolean backpropAll;
	
	// current error
	protected float error = 0;
	
	public BpttProcessor(TensorFactory factory, 
			NeuralNetwork nn, 
			DataLogger logger, 
			SequenceDataset d, 
			SamplingStrategy s,
			Criterion c,
			int sequenceLength,
			boolean backpropAll) {
		super(factory, nn, logger);
	
		this.dataset = d;
		this.sampling = s;
		this.criterion = c;
		
		this.sequenceLength = sequenceLength;
		this.backpropAll = backpropAll;
	}
	
	
	@Override
	public float processNext(float f) {
		error = 0;

		int index = sampling.next();
		if(dataset.size()-index < sequenceLength+1){
			index-=(sequenceLength+1);
			if(index < 0){
				throw new RuntimeException("Sequence length larger than dataset...");
			}
		}
		
		Tensor[] sequence = dataset.getSequence(index, sequenceLength);
		
		// keep all memories intermediate states
		Map<UUID, Tensor[]> memories = new HashMap<UUID, Tensor[]>();
		nn.getMemories().entrySet().forEach(e -> {
			Tensor[] mems = new Tensor[sequenceLength];
			for(int i=0;i<sequenceLength;i++){
				mems[i] = e.getValue().getMemory().copyInto(null);
			}
			memories.put(e.getKey(), mems);
		});
		
		// also keep all outputs (in case we want to backprop all)
		Tensor[] outputs = new Tensor[sequenceLength];
		
		
		for(int i=0;i<sequenceLength;i++){
			final int in = i;
			nn.getMemories().entrySet().forEach(e ->{
				e.getValue().getMemory().copyInto(memories.get(e.getKey())[in]);
			});
				
			// forward
			Tensor out = nn.forward(sequence[i]);
				
			// store output
			outputs[i] = out;
		}
		
		
		// backward
		for(int i=sequenceLength-1;i>=0;i--){
			Tensor target = sequence[i+1];
			float err = criterion.error(outputs[i], target).get(0);
			error+=err;
			Tensor grad = criterion.grad(outputs[i], target);
				
			// first forward again with correct state and memories
			final int in = i;
			nn.getMemories().entrySet().forEach(e -> {
				e.getValue().setMemory(memories.get(e.getKey())[in]);
			});
			nn.forward(sequence[i]);
				
			// set grad to zero for all intermediate outputs
			if(!backpropAll){
				if(i!=sequenceLength-1){
					grad.fill(0.0f);
				}
			}
			nn.backward(grad);
				
			// acc grad
			nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
				
		}
		
		return error;
	}
}
