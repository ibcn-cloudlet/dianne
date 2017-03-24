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
package be.iminds.iot.dianne.nn.learn.strategy;

import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.BatchSampler;
import be.iminds.iot.dianne.nn.learn.strategy.config.FeedForwardConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Default LearningStrategy for supervised training of a Neural Network
 * 
 * @author tverbele
 *
 */
public class FeedForwardLearningStrategy implements LearningStrategy {

	protected Dataset dataset;
	protected NeuralNetwork nn;
	
	protected FeedForwardConfig config;
	protected GradientProcessor gradientProcessor;
	protected Criterion criterion;
	protected BatchSampler sampler;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		this.dataset = dataset;
		this.nn = nns[0];
		
		// Store the labels if classification dataset
		String[] labels = dataset.getLabels();
		if(labels!=null)
			nn.setOutputLabels(labels);
		
		this.config = DianneConfigHandler.getConfig(config, FeedForwardConfig.class);
		
		sampler = new BatchSampler(dataset, this.config.sampling, config);
		criterion = CriterionFactory.createCriterion(this.config.criterion, config);
		gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, nn, config);
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Clear delta params
		nn.zeroDeltaParameters();

		// Load batch - reuse memory
		Batch batch = sampler.nextBatch();

		// Forward input
		Tensor output = nn.forward(batch.input);
		
		// Calculate loss
		float loss = TensorOps.mean(criterion.loss(output, batch.target));
		
		// Calculate gradient on the outputs
		Tensor gradOutput = criterion.grad(output, batch.target);
		
		// Backpropagate
		nn.backward(gradOutput);

		// Accumulate gradients in delta params
		nn.accGradParameters();

		// Run gradient processors
		gradientProcessor.calculateDelta(i);

		// Update parameters
		nn.updateParameters();

		return new LearnProgress(i, loss);
	}

}
