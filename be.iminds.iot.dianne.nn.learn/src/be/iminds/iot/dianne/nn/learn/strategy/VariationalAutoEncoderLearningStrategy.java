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
 *     Steven Bohez
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
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.BatchConfig;
import be.iminds.iot.dianne.nn.learn.criterion.GaussianKLDivCriterion;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.BatchSampler;
import be.iminds.iot.dianne.nn.learn.strategy.config.FeedForwardConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Strategy for learning Variational Auto Encoders (VAEs). Assumes factorized Gaussian-distributed latent variables.
 * Also assumes 1D encoder output (2D for batches).
 * 
 * @author smbohez
 *
 */
public class VariationalAutoEncoderLearningStrategy implements LearningStrategy {
	
	protected FeedForwardConfig config;
	
	protected int latentDims = 1;
	protected int sampleSize = 1;
	
	protected Dataset dataset;
	protected BatchSampler sampler;
	
	protected NeuralNetwork encoder;
	protected NeuralNetwork decoder;
	
	protected GradientProcessor encoderProcessor;
	protected GradientProcessor decoderProcessor;
	
	protected Tensor prior;
	protected Tensor random;
	
	protected Tensor latentParams;
	protected Tensor latentParamsGrad;
	
	protected Tensor latent;
	protected Tensor latentGrad;
	
	protected Criterion reconCriterion;
	protected Criterion regulCriterion;

	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		this.dataset = dataset;
		
		this.encoder = nns[0];
		this.decoder = nns[1];
		
		if(config.containsKey("latentDims"))
			this.latentDims = Integer.parseInt(config.get("latentDims"));
		
		if(config.containsKey("sampleSize"))
			this.sampleSize = Integer.parseInt(config.get("sampleSize"));

		this.config = DianneConfigHandler.getConfig(config, FeedForwardConfig.class);
		this.sampler = new BatchSampler(dataset, this.config.sampling, config);
		
		this.encoderProcessor = ProcessorFactory.createGradientProcessor(this.config.method, this.encoder, config);
		this.decoderProcessor = ProcessorFactory.createGradientProcessor(this.config.method, this.decoder, config);
		
		// Set criteria
		// TODO: set regularization criterion based on modeled distributions
		this.reconCriterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.regulCriterion = new GaussianKLDivCriterion(DianneConfigHandler.getConfig(config, BatchConfig.class));
		
		// Set prior distribution parameters = standard normal
		this.prior = new Tensor(this.config.batchSize, 2*this.latentDims);
		this.prior.narrow(1, 0, this.latentDims).fill(0);
		this.prior.narrow(1, this.latentDims, this.latentDims).fill(1);
		
		// Preallocate some Tensors for efficiency
		this.random = new Tensor(this.config.batchSize, this.latentDims);
		this.latent = new Tensor(this.config.batchSize, this.latentDims);
		this.latentParamsGrad = new Tensor(this.config.batchSize, 2*this.latentDims);
	}
	
	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Clear delta params
		encoder.zeroDeltaParameters();
		decoder.zeroDeltaParameters();
		latentParamsGrad.fill(0);
		
		// Load input batch
		Batch batch = sampler.nextBatch();
		
		// Get latent distribution parameters (encoder forward)
		latentParams = encoder.forward(batch.input);
		
		// For a number of latent samples...
		// TODO: unfold loop over samples into single batch
		float reconstructionError = 0;
		for(int s = 0; s < sampleSize; s++) {
			// Sample latent variables (latentParams => latent)
			sampleLatentVariables();
			
			// Get output distribution parameters (decoder forward)
			Tensor outputParams = decoder.forward(latent);
			
			// Reconstruction error & gradient on decoder (outputParams => reconstructionError,reconstructionGrad)
			// Note: scaling here is easier than outside this loop
			reconstructionError += TensorOps.mean(reconCriterion.loss(outputParams, batch.target))/sampleSize;
			Tensor reconstructionGrad = reconCriterion.grad(outputParams, batch.target);
			TensorOps.div(reconstructionGrad, reconstructionGrad, sampleSize);
			
			// Get gradient on latent variables (decoder backward)
			latentGrad = decoder.backward(reconstructionGrad);
			
			// Accumulate gradients in decoder
			decoder.accGradParameters();
			
			// Add gradient on latent params (latentGrad => latentParamsGrad)
			accLatentParamsGradient();
		}
		
		// Regularization error & gradient (latentParams => regularizationError,regularizationGrad)
		float regularizationError = TensorOps.mean(regulCriterion.loss(latentParams, prior));
		Tensor regularizationGrad = regulCriterion.grad(latentParams, prior);
		
		// Add to latent gradient
		TensorOps.add(latentParamsGrad, latentParamsGrad, regularizationGrad);
		
		// Encoder backward
		encoder.backward(latentParamsGrad);
		
		// Accumulate gradients in delta params
		encoder.accGradParameters();

		// Run gradient processors
		encoderProcessor.calculateDelta(i);
		decoderProcessor.calculateDelta(i);
		
		// Update parameters
		encoder.updateParameters();
		decoder.updateParameters();

		return new LearnProgress(i, reconstructionError+regularizationError);
	}
	
	private void sampleLatentVariables() {
		// latentParam => latent
		Tensor means = latentParams.narrow(1, 0, latentDims);
		Tensor stdevs = latentParams.narrow(1, latentDims, latentDims);
		
		random.randn();
		
		TensorOps.cmul(latent, random, stdevs);
		TensorOps.add(latent, latent, means);
	}
	
	private void accLatentParamsGradient() {
		// latentGrad => latentParamsGrad
		Tensor gradMeans = latentParamsGrad.narrow(1, 0, latentDims);
		Tensor gradStdevs = latentParamsGrad.narrow(1, latentDims, latentDims);
		
		TensorOps.add(gradMeans, gradMeans, latentGrad);
		TensorOps.addcmul(gradStdevs, gradStdevs, 1, latentGrad, random);
	}
	
}
