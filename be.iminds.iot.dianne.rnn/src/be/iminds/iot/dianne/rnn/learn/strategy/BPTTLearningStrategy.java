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
package be.iminds.iot.dianne.rnn.learn.strategy;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rnn.dataset.SequenceDataset;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rnn.learn.strategy.config.BPTTConfig;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This LearningStrategy works on SequenceDatasets and trains according to the
 * Back Propagate Through Time (BPTT) principle.
 * 
 * @author tverbele
 *
 */
public class BPTTLearningStrategy implements LearningStrategy {

	protected SequenceDataset dataset;
	protected NeuralNetwork nn;
	
	protected BPTTConfig config;
	protected GradientProcessor gradientProcessor;
	protected Criterion criterion;
	protected SamplingStrategy sampling;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof SequenceDataset))
			throw new RuntimeException("Dataset is no sequence dataset");
		
		this.dataset = (SequenceDataset)dataset;
		this.nn = nns[0];
		
		this.config = DianneConfigHandler.getConfig(config, BPTTConfig.class);
		sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		criterion = CriterionFactory.createCriterion(this.config.criterion, config);
		gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, nn, config);
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// clear delta params
		nn.zeroDeltaParameters();
		
		// calculate grad through sequence
		float loss = 0;
		int index = sampling.next();
		if(dataset.size()-index < config.sequenceLength+1){
			index-=(config.sequenceLength+1);
			if(index < 0){
				throw new RuntimeException("Sequence length larger than dataset...");
			}
		}
		
		Tensor[] sequence = dataset.getSequence(index, config.sequenceLength);
		
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
			float l = criterion.loss(outputs[k], target);
			if(config.backpropAll || k==config.sequenceLength-1){
				loss+=l;
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
			nn.accGradParameters();
				
		}
		
		// run gradient processors
		gradientProcessor.calculateDelta(i);
		
		// update parameters
		nn.updateParameters();
		
		return new LearnProgress(i, loss);
	}

}
