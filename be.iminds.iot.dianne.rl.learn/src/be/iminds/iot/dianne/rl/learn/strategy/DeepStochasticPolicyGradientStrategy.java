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
package be.iminds.iot.dianne.rl.learn.strategy;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.BatchConfig;
import be.iminds.iot.dianne.nn.learn.criterion.GaussianKLDivCriterion;
import be.iminds.iot.dianne.nn.learn.criterion.PseudoHuberCriterion;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.DeepStochasticPolicyGradientConfig;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class DeepStochasticPolicyGradientStrategy implements LearningStrategy {

	protected DeepStochasticPolicyGradientConfig config;
	
	protected ExperiencePool pool;
	protected SamplingStrategy experienceSampling;
	protected ExperiencePoolBatch batch;
	
	protected NeuralNetwork actor;
	protected NeuralNetwork targetActor;
	
	protected NeuralNetwork critic;
	protected NeuralNetwork targetCritic;
	
	protected Criterion reconCriterion;
	protected Criterion criticRegulCriterion;
	protected Criterion actorRegulCriterion;
	
	protected GradientProcessor actorProcessor;
	protected GradientProcessor criticProcessor;
	
	protected UUID stateIn;
	protected UUID actionIn;
	protected UUID valueOut;
	
	protected UUID[] inputIds;
	protected UUID[] outputIds;
	
	protected Tensor random;
	protected Tensor actionPrior;
	protected Tensor actionSample;
	
	protected Tensor targetValue;
	
	protected Tensor criticGrad;
	protected Tensor actorGrad;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset is no experience pool");
		
		this.pool = (ExperiencePool) dataset;
		
		if(nns.length != 4)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 4)");
		
		this.actor = nns[0];
		this.targetActor = nns[1];
		this.critic = nns[2];
		this.targetCritic = nns[3];
		
		this.config = DianneConfigHandler.getConfig(config, DeepStochasticPolicyGradientConfig.class);
		this.experienceSampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		
		this.criticProcessor = ProcessorFactory.createGradientProcessor(this.config.method, critic, config);
		config.put("learningRate", String.valueOf(Float.parseFloat(config.get("learningRate"))*this.config.policyRateScaling));
		this.actorProcessor = ProcessorFactory.createGradientProcessor(this.config.method, actor, config);
		
		// Look for the critic inputs corresponding to state & action
		NeuralNetworkInstanceDTO nndto = this.critic.getNeuralNetworkInstance();
		for(UUID iid : this.critic.getInputs().keySet()) {
			ModuleInstanceDTO mdto = nndto.modules.get(iid);
			String mname = mdto.module.properties.get("name");
			
			if(mname.equalsIgnoreCase("state"))
				this.stateIn = iid;
			else if(mname.equalsIgnoreCase("action"))
				this.actionIn = iid;
		}
		this.valueOut = this.critic.getOutput().getId();
		
		if(stateIn == null || actionIn == null || valueOut == null)
			throw new RuntimeException("Unable to select correct Input modules from network " + nndto.name);
		
		this.inputIds = new UUID[]{this.stateIn, this.actionIn};
		this.outputIds = new UUID[]{this.valueOut};
		
		this.reconCriterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.criticRegulCriterion = new PseudoHuberCriterion(DianneConfigHandler.getConfig(config, BatchConfig.class));
		this.actorRegulCriterion = new GaussianKLDivCriterion(DianneConfigHandler.getConfig(config, BatchConfig.class));
		
		// Pre-allocate tensors for batch operations
		this.random = new Tensor(this.config.batchSize, this.pool.actionDims());
		this.actionSample = new Tensor(this.config.batchSize, this.pool.actionDims());
		
		// TODO: set tensors based on modeled distributions, currently a 1D factorized Gaussian is assumed
		this.actionPrior = new Tensor(this.config.batchSize, this.pool.actionDims()[0]*2);
		this.actionPrior.narrow(1, 0, this.pool.actionDims()[0]).fill(0);
		this.actionPrior.narrow(1, this.pool.actionDims()[0], this.pool.actionDims()[0]).fill(this.config.actionPriorDev);
		
		this.targetValue = new Tensor(this.config.batchSize);
		
		this.criticGrad = new Tensor(this.config.batchSize, 1);
		this.actorGrad = new Tensor(this.config.batchSize, this.pool.actionDims()[0]*2);
		
		// Wait for the pool to contain enough samples
		if(pool.size() < this.config.minSamples){
			System.out.println("Experience pool has too few samples, waiting a bit to start learning...");
			while(pool.size() < this.config.minSamples){
				try {
					Thread.sleep(5000);
				} catch (InterruptedException e) {
					return;
				}
			}
		}
		System.out.println("Start learning...");
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Reset the deltas
		actor.zeroDeltaParameters();
		critic.zeroDeltaParameters();
		criticGrad.fill(0);
		actorGrad.fill(0);
		
		// Fill in the batch
		batch = pool.getBatch(batch, experienceSampling.next(config.batchSize));
		
		// Get distribution parameters for action in next state using target actor
		Tensor nextActionParams = targetActor.forward(batch.getNextState());
		
		// Get the value estimate for the current state using the working critic
		Tensor value = critic.forward(inputIds, outputIds, new Tensor[]{batch.getState(), batch.getAction()}).getValue().tensor;
		
		// Generate a number of samples to update the critic
		float loss = 0;
		for(int s = 0; s < config.criticSamples; s++) {
			// Sample action in next state
			sampleAction(actionSample, random, nextActionParams);
			
			// Calculate target values for sampled action
			Tensor nextValue = targetCritic.forward(inputIds, outputIds, new Tensor[]{batch.getNextState(), actionSample}).getValue().tensor;
			targetValue = TensorOps.addcmul(targetValue, batch.getReward(), config.discount, batch.getTerminal(), nextValue);

			// Calculate the loss and gradient with respect to the target value
			// Note: scaling here is easier than outside this loop
			// TODO: calculate importance weight?
			loss += TensorOps.mean(reconCriterion.loss(value, targetValue))/config.criticSamples;
			TensorOps.add(criticGrad, criticGrad, 1f/config.criticSamples, reconCriterion.grad(value, targetValue));
		}
		
		// Add value smoothing on critic gradient when required
		if(config.smoothingRegularization > 0) {
			targetValue.fill(TensorOps.mean(value));
			
			loss += config.smoothingRegularization*TensorOps.mean(criticRegulCriterion.loss(value, targetValue));
			TensorOps.add(criticGrad, criticGrad, config.smoothingRegularization, criticRegulCriterion.grad(value, targetValue));
		}
		
		// Backward pass of the critic
		critic.backward(outputIds, inputIds, new Tensor[]{criticGrad}).getValue();
		critic.accGradParameters();
		
		// Set the critic gradient in order to update the actor
		// Note: by default we're doing minimization, so critic gradient is set to -1/#samples
		criticGrad.fill(-1f/(config.batchSize*config.actorSamples));
		
		// Get distribution parameters for action in current state using working actor
		Tensor actionParams = actor.forward(batch.getState());
		
		// Generate a number of samples to update the actor
		for(int s = 0; s < config.actorSamples; s++) {
			// Sample action in the current state
			sampleAction(actionSample, random, actionParams);
			
			// Get the actor gradient by evaluating the critic and use it's gradient with respect to the action
			value = critic.forward(inputIds, outputIds, new Tensor[]{batch.getState(), actionSample}).getValue().tensor;
			Tensor actionGrad = critic.backward(outputIds, inputIds, new Tensor[]{criticGrad}).getValue().tensors.get(actionIn);
			
			// Accumulate gradient on action parameters
			accActionParamsGradient(actorGrad, random, actionGrad);
		}
		
		// Add action prior regularization error and gradient on action parameters
		if(config.actionPriorRegularization > 0) {
			loss += config.actionPriorRegularization*TensorOps.mean(actorRegulCriterion.loss(actionParams, actionPrior));
			TensorOps.add(actorGrad, actorGrad, config.actionPriorRegularization, actorRegulCriterion.grad(actionParams, actionPrior));
		}
		
		// Add trust region regularization error and gradient on action parameters
		if(config.trustRegionRegularization > 0) {
			Tensor targetActionParams = targetActor.forward(batch.getState());
			
			loss += config.trustRegionRegularization*TensorOps.mean(actorRegulCriterion.loss(actionParams, targetActionParams));
			TensorOps.add(actorGrad, actorGrad, config.trustRegionRegularization, actorRegulCriterion.grad(actionParams, targetActionParams));
		}
		
		// Perform gradient clipping if required
		if(config.actorGradClipping) {
			ModuleOps.tanh(actorGrad, actorGrad);
		}
		
		// Backward pass of the actor
		actor.backward(actorGrad);
		actor.accGradParameters();
		
		// Call the processors to set the updates
		actorProcessor.calculateDelta(i);
		criticProcessor.calculateDelta(i);
		
		// Apply the updates
		// Note: target actor & critic get updated automatically by setting the syncInterval option
		actor.updateParameters();
		critic.updateParameters();
		
		// Report the average loss and value of the current policy
		// Note: currently this is the value of the last actor sample only
		return new LearnProgress(i, loss, new String[]{"q"}, new float[]{ TensorOps.sum(value)/config.batchSize});
	}

	private static void sampleAction(Tensor action, Tensor random, Tensor actionParams) {
		int actionDims = action.size(1);
		
		Tensor means = actionParams.narrow(1, 0, actionDims);
		Tensor stdevs = actionParams.narrow(1, actionDims, actionDims);
		
		random.randn();
		
		TensorOps.cmul(action, random, stdevs);
		TensorOps.add(action, action, means);
	}
	
	private static void accActionParamsGradient(Tensor actionParamsGrad, Tensor random, Tensor actionGrad) {
		int actionDims = actionGrad.size(1);
		
		Tensor gradMeans = actionParamsGrad.narrow(1, 0, actionDims);
		Tensor gradStdevs = actionParamsGrad.narrow(1, actionDims, actionDims);
		
		TensorOps.add(gradMeans, gradMeans, actionGrad);
		TensorOps.addcmul(gradStdevs, gradStdevs, 1, actionGrad, random);
	}
}
