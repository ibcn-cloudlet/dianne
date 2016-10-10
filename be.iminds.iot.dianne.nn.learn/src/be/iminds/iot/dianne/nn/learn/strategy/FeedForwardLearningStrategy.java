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

import java.util.Arrays;
import java.util.Map;

import org.osgi.util.promise.Promise;

import be.iminds.iot.dianne.api.dataset.Batch;
import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
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
	protected SamplingStrategy sampling;
	
	protected Batch batch = null;
	private Batch nextBatch = null;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		this.dataset = dataset;
		this.nn = nns[0];
		
		// Store the labels if classification dataset
		String[] labels = dataset.getLabels();
		if(labels!=null)
			nn.setOutputLabels(labels);
		
		this.config = DianneConfigHandler.getConfig(config, FeedForwardConfig.class);
		sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		criterion = CriterionFactory.createCriterion(this.config.criterion);
		gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, nn, config);
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Clear delta params
		nn.zeroDeltaParameters();
		
		// Use a placeholder for error
		final float[] error = new float[1];
		
		// Load batch for first iteration
		if(nextBatch==null)
			nextBatch = dataset.getBatch(nextBatch, sampling.next(config.batchSize));
		
		// Flip current/next
		Batch temp = batch;
		batch = nextBatch;
		nextBatch = temp;
		
		// Forward/backward pass - executed asynchronously
		// Handle case of varying input dims so batch is array of different tensors
		// Load next batch while doing forward/backward
		if(batch.input != null) {
			// Execute in batch
			Promise result = nn.forward(null, null, batch.input).then(
					p -> {
						// Forward
						Tensor output = p.getValue().tensor;
						
						// Error
						error[0] += criterion.loss(output, batch.target).get(0);
	
						// Error gradient
						Tensor gradOut = criterion.grad(output, batch.target);
						
						// Backward
						return nn.backward(null, null, gradOut);
					}).then(		
					p -> {	
						// Accumulate gradient weights
						nn.getTrainables().values().stream().forEach(Trainable::accGradParameters);
						
						return p;
					});
			
			// Load next batch while processing previous one
			nextBatch = dataset.getBatch(nextBatch, sampling.next(config.batchSize));
			
			// Fetch the result (errors are handled by caller)
			result.getValue();
		} else {
			// Cannot load a batch for this dataset, still process one by one
			for(int k=0;k<config.batchSize;k++){
				final int b = k;
				Promise result = nn.forward(null, null, batch.samples[b].input).then(
						p -> {
							// Forward
							Tensor output = p.getValue().tensor;
							
							// Error
							error[0] += criterion.loss(output, batch.samples[b].target).get(0);
		
							// Error gradient
							Tensor gradOut = criterion.grad(output, batch.samples[b].target);
							
							// Backward
							return nn.backward(null, null, gradOut);
						}).then(
						p -> {	
							// Accumulate gradient weights
							nn.getTrainables().values().stream().forEach(Trainable::accGradParameters);
							
							return p;
						});
				
				//Fetch the result (errors are handled by caller)
				result.getValue();
			}
			
			// Load next batch
			nextBatch = dataset.getBatch(nextBatch, sampling.next(config.batchSize));
		}

		// Batch done, calculate deltas
		if(config.batchAverage) {
			// Divide by batchSize in order to have learning rate independent of batchSize
			nn.getTrainables().values().stream().forEach(m -> {
				Tensor deltaParams = m.getDeltaParameters();
	
				TensorOps.div(deltaParams, deltaParams, config.batchSize);
		
				// Set DeltaParameters to be sure in case of remote module instance
				m.setDeltaParameters(deltaParams);
			});
			
			error[0] /= config.batchSize;
		}
		
		// Run gradient processors
		gradientProcessor.calculateDelta(i);
		
		// Update parameters
		nn.updateParameters();
		
		return new LearnProgress(i, error[0]);		
	}

}
