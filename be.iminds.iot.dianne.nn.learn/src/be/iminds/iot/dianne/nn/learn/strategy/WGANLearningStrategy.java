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
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.BatchSampler;
import be.iminds.iot.dianne.nn.learn.strategy.config.GenerativeAdverserialConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Wasserstein GAN Learning strategy
 * 
 * https://arxiv.org/abs/1701.07875
 * 
 * @author tverbele
 *
 */
public class WGANLearningStrategy implements LearningStrategy {

	protected Dataset dataset;
	
	protected NeuralNetwork generator;
	protected NeuralNetwork discriminator;
	
	protected GenerativeAdverserialConfig config;
	
	protected GradientProcessor gradientProcessorG;
	protected GradientProcessor gradientProcessorD;

	protected Criterion criterion;
	protected BatchSampler sampler;
	
	protected Tensor grad;
	protected Tensor random;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		this.dataset = dataset;
		
		this.generator = nns[0];
		this.discriminator = nns[1];
		
		this.config = DianneConfigHandler.getConfig(config, GenerativeAdverserialConfig.class);

		sampler = new BatchSampler(dataset, this.config.sampling, config);

		gradientProcessorG = ProcessorFactory.createGradientProcessor(this.config.method, generator, config);
		gradientProcessorD = ProcessorFactory.createGradientProcessor(this.config.method, discriminator, config);
		
		grad = new Tensor(this.config.batchSize, 1);
		random = new Tensor(this.config.batchSize, this.config.generatorDim);
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Clear delta params
		generator.zeroDeltaParameters();
		discriminator.zeroDeltaParameters();
		
		// First update the discriminator
		
		// Load minibatch of real data for the discriminator 
		Batch batch = sampler.nextBatch();
		
		
		Tensor output = discriminator.forward(batch.input);
		float d_loss_positive = TensorOps.mean(output);
		grad.fill(-1.0f/this.config.batchSize);
		discriminator.backward(grad);
		
		// Keep gradients to the parameters
		discriminator.accGradParameters();
		
		// Now sample a minibatch of generated data
		random.randn();

		Tensor generated = generator.forward(random);
		output = discriminator.forward(generated);
		float d_loss_negative = TensorOps.mean(output);
		grad.fill(1.0f/this.config.batchSize);
		Tensor gradInput = discriminator.backward(grad);
		
		// Update discriminator weights
		discriminator.accGradParameters();
		
		// Clamp discriminator weights
		discriminator.getParameters().values().forEach(t -> TensorOps.clamp(t, t, -0.01f, 0.01f));
		
		// Run gradient processors
		gradientProcessorD.calculateDelta(i);
		
		discriminator.updateParameters();
		
		// Now update the generator
		// Just flip the sign of the gradIn of discriminator backward 
		TensorOps.mul(gradInput, gradInput, -1.0f);
		generator.backward(gradInput);
		
		// Update generator weights
		generator.accGradParameters();
		
		// Run gradient processors
		gradientProcessorG.calculateDelta(i);
		
		// Update parameters
		generator.updateParameters();
		
		
		return new LearnProgress(i, -( d_loss_positive - d_loss_negative));
	}

}
