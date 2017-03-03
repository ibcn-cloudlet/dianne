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
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This strategy uses two neural networks, the first one being the trainee and the second being the target.
 * 
 * For each sample of the dataset the loss is based on the criterion between the trainee output and the target output.
 * 
 * @author tverbele
 *
 */
public class TargetLearningStrategy implements LearningStrategy {

	protected FeedForwardConfig config;
	
	protected Dataset dataset;
	protected SamplingStrategy sampling;
	
	protected Batch batch;
	
	protected NeuralNetwork traineeNetwork;
	protected NeuralNetwork targetNetwork;
	
	protected Criterion criterion;
	protected GradientProcessor gradientProcessor;

	protected Tensor output;
	protected Tensor target;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		this.dataset = dataset;
		
		if(nns.length != 2)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 2)");
			
		this.traineeNetwork = nns[0];
		this.targetNetwork = nns[1];
		
		this.config = DianneConfigHandler.getConfig(config, FeedForwardConfig.class);
		this.sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		this.criterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, traineeNetwork, config);
		
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Reset the deltas
		traineeNetwork.zeroDeltaParameters();
		
		// Fill in the batch
		batch = dataset.getBatch(batch, sampling.next(config.batchSize));
		
		// Forward pass 
		output = traineeNetwork.forward(batch.input);
		target = targetNetwork.forward(batch.input);
		
		
		Tensor l = criterion.loss(output, target);
		float loss = TensorOps.mean(l);
		Tensor grad = criterion.grad(output, target);
		
		// Backward pass of the critic
		traineeNetwork.backward(grad);
		traineeNetwork.accGradParameters();
		
		// Call the processors to set the updates
		gradientProcessor.calculateDelta(i);
		
		// Apply the updates
		// Note: target network gets updated automatically by setting the syncInterval option
		traineeNetwork.updateParameters();
		
		return new LearnProgress(i, loss);
	}

}
