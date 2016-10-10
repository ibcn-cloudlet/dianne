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
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.learn.strategy.config.FeedForwardConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;

public class SimpleLearningStrategy implements LearningStrategy {

	protected Dataset dataset;
	protected NeuralNetwork nn;
	
	protected FeedForwardConfig config;
	protected GradientProcessor gradientProcessor;
	protected Criterion criterion;
	protected SamplingStrategy sampling;
	
	protected Batch batch;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		this.dataset = dataset;
		this.nn = nns[0];
		
		this.config = DianneConfigHandler.getConfig(config, FeedForwardConfig.class);
		sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		criterion = CriterionFactory.createCriterion(this.config.criterion, config);
		gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, nn, config);
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Clear delta params
		nn.zeroDeltaParameters();
		
		// Load batch - reuse memory 
		batch = dataset.getBatch(batch, sampling.next(this.config.batchSize));
		
		// Forward-backward pass
		Tensor output = nn.forward(batch.input);
		float error = criterion.loss(output, batch.target).get(0);
		Tensor gradOutput = criterion.grad(output, batch.target);
		nn.backward(gradOutput);
		
		// Accumulate gradients in delta params
		nn.accGradParameters();
		
		// Run gradient processors
		gradientProcessor.calculateDelta(i);
		
		// Update parameters
		nn.updateParameters();
		
		return new LearnProgress(i, error);
	}

}
