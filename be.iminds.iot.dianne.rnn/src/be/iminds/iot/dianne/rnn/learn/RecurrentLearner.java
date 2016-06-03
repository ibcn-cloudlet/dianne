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
package be.iminds.iot.dianne.rnn.learn;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.learn.Learner;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.api.rnn.dataset.SequenceDataset;
import be.iminds.iot.dianne.nn.learn.AbstractLearner;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rnn.learn.config.RecurrentLearnerConfig;
import be.iminds.iot.dianne.tensor.Tensor;

@Component(service=Learner.class, 
	property={"aiolos.unique=true",
			  "dianne.learner.category=RNN"})
public class RecurrentLearner extends AbstractLearner {
	
	protected RecurrentLearnerConfig config;

	protected void loadConfig(Map<String, String> config){
		super.loadConfig(config);
		
		this.config = DianneConfigHandler.getConfig(config, RecurrentLearnerConfig.class);
	}
	
	protected float process(long i){
		// clear delta params
		nn.getTrainables().entrySet().stream().forEach(e -> {
			e.getValue().zeroDeltaParameters();
		});
		
		// calculate grad through sequence
		float err = 0;
		int index = sampling.next();
		if(dataset.size()-index < config.sequenceLength+1){
			index-=(config.sequenceLength+1);
			if(index < 0){
				throw new RuntimeException("Sequence length larger than dataset...");
			}
		}
		
		Tensor[] sequence = ((SequenceDataset)dataset).getSequence(index, config.sequenceLength);
		
		// keep all memories hidden states
		Map<UUID, Tensor[]> memories = new HashMap<UUID, Tensor[]>();
		nn.getMemories().entrySet().forEach(e -> {
			Tensor[] mems = new Tensor[config.sequenceLength];
			for(int k=0;k<config.sequenceLength;k++){
				mems[k] = e.getValue().getMemory().copyInto(null);
			}
			memories.put(e.getKey(), mems);
		});
		
		// also keep all outputs (in case we want to backprop all)
		Tensor[] outputs = new Tensor[config.sequenceLength];
		
		
		for(int k=0;k<config.sequenceLength;k++){
			final int in = k;
			nn.getMemories().entrySet().forEach(e ->{
				e.getValue().getMemory().copyInto(memories.get(e.getKey())[in]);
			});
				
			// forward
			Tensor out = nn.forward(sequence[k]);
				
			// store output
			outputs[k] = out;
		}
		
		
		// backward
		for(int k=config.sequenceLength-1;k>=0;k--){
			Tensor target = sequence[k+1];
			float er = criterion.error(outputs[k], target).get(0);
			if(config.backpropAll || k==config.sequenceLength-1){
				err+=er;
			}
			Tensor grad = criterion.grad(outputs[k], target);
				
			// first forward again with correct state and memories
			final int in = k;
			nn.getMemories().entrySet().forEach(e -> {
				e.getValue().setMemory(memories.get(e.getKey())[in]);
			});
			nn.forward(sequence[k]);
				
			// set grad to zero for all intermediate outputs
			if(!config.backpropAll){
				if(k!=config.sequenceLength-1){
					grad.fill(0.0f);
				}
			}
			nn.backward(grad);
				
			// acc grad
			nn.getTrainables().values().stream().forEach(m -> m.accGradParameters());
				
		}
		
		
		// run gradient processors
		gradientProcessor.calculateDelta(i);
		
		// update parameters
		nn.getTrainables().values().stream().forEach(Trainable::updateParameters);
		
		return err;
	}
	
}
