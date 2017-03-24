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

import java.util.List;
import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.dataset.SequenceDataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rnn.learn.criterion.SequenceCriterion;
import be.iminds.iot.dianne.rnn.learn.criterion.SequenceCriterionFactory;
import be.iminds.iot.dianne.rnn.learn.sampling.SequenceSampler;
import be.iminds.iot.dianne.rnn.learn.strategy.config.BPTTConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This LearningStrategy works on SequenceDatasets and trains according to the
 * Back Propagate Through Time (BPTT) principle.
 * 
 * @author tverbele
 *
 */
@SuppressWarnings("rawtypes")
public class BPTTLearningStrategy implements LearningStrategy {

	protected SequenceDataset dataset;
	protected NeuralNetwork nn;
	
	protected BPTTConfig config;
	protected GradientProcessor gradientProcessor;
	protected SequenceCriterion criterion;
	protected SequenceSampler sampler;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof SequenceDataset))
			throw new RuntimeException("Dataset is no sequence dataset");
		
		this.dataset = (SequenceDataset)dataset;
		this.nn = nns[0];
		
		String[] labels = this.dataset.getLabels();
		if(labels!=null)
			nn.setOutputLabels(labels);
		
		this.config = DianneConfigHandler.getConfig(config, BPTTConfig.class);
		sampler = new SequenceSampler(this.dataset, this.config.sampling, config);
		criterion = SequenceCriterionFactory.createCriterion(this.config.criterion, config);
		gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, nn, config);
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// clear delta params
		nn.zeroDeltaParameters();
		// reset memory
		this.nn.resetMemory(this.config.batchSize);

		// sample sequence
		Sequence<Batch> sequence = sampler.nextSequence();
		
		// forward
		List<Tensor> outputs = nn.forward(sequence.getInputs());
		
		// calculate gradients
		List<Tensor> targets = sequence.getTargets();
		float loss =  TensorOps.mean(criterion.loss(outputs, targets).stream().reduce((t1,t2) -> TensorOps.add(t1, t1, t2)).get())/sequence.size;
		List<Tensor> gradOutputs = criterion.grad(outputs, targets);
		
		// backward and acc grad parameters
		nn.backward(gradOutputs, true);

		// run gradient processors
		gradientProcessor.calculateDelta(i);
		
		// update parameters
		nn.updateParameters();
		
		return new LearnProgress(i, loss);
	}

}
