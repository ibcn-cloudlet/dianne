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
import be.iminds.iot.dianne.nn.learn.criterion.PseudoHuberCriterion;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.DeepDeterministicPolicyGradientConfig;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class DeepDeterministicPolicyGradientStrategy implements LearningStrategy {

	protected DeepDeterministicPolicyGradientConfig config;
	
	protected ExperiencePool pool;
	protected SamplingStrategy sampling;
	protected ExperiencePoolBatch batch;
	
	protected NeuralNetwork actor;
	protected NeuralNetwork targetActor;
	
	protected NeuralNetwork critic;
	protected NeuralNetwork targetCritic;
	
	protected Criterion reconCriterion;
	protected Criterion regulCriterion;
	protected GradientProcessor actorProcessor;
	protected GradientProcessor criticProcessor;
	
	protected UUID stateIn;
	protected UUID actionIn;
	protected UUID valueOut;
	
	protected UUID[] inputIds;
	protected UUID[] outputIds;
	
	protected Tensor targetValue;
	
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
		
		this.config = DianneConfigHandler.getConfig(config, DeepDeterministicPolicyGradientConfig.class);
		this.sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		this.reconCriterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.regulCriterion = new PseudoHuberCriterion(DianneConfigHandler.getConfig(config, BatchConfig.class));
		
		config.put("learningRate", String.valueOf(this.config.learningRate));
		this.criticProcessor = ProcessorFactory.createGradientProcessor(this.config.method, critic, config);
		config.put("learningRate", String.valueOf(this.config.learningRate*this.config.policyRateScaling));
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
		
		// Pre-allocate tensors for batch operations
		this.targetValue = new Tensor(this.config.batchSize);
		
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
		
		// Fill in the batch
		batch = pool.getBatch(batch, sampling.next(config.batchSize));
		
		// Calculate targetValues
		Tensor nextAction = targetActor.forward(batch.getNextState());
		Tensor nextValue = targetCritic.forward(inputIds, outputIds, new Tensor[]{batch.getNextState(), nextAction}).getValue().tensor;
		targetValue = TensorOps.addcmul(targetValue, batch.getReward(), config.discount, batch.getTerminal(), nextValue);
		
		// Forward pass of the critic to get the current value estimate
		Tensor value = critic.forward(inputIds, outputIds, new Tensor[]{batch.getState(), batch.getAction()}).getValue().tensor;
		
		// Calculate the loss and gradient with respect to the target value
		float loss = TensorOps.mean(reconCriterion.loss(value, targetValue));
		Tensor criticGrad = reconCriterion.grad(value, targetValue);
		
		// Add value smoothing on critic gradient when required
		if(config.smoothingRegularization > 0) {
			targetValue.fill(TensorOps.mean(value));
			
			loss += config.smoothingRegularization*TensorOps.mean(regulCriterion.loss(value, targetValue));
			TensorOps.add(criticGrad, criticGrad, config.smoothingRegularization, regulCriterion.grad(value, targetValue));
		}
		
		// Backward pass of the critic
		critic.backward(outputIds, inputIds, new Tensor[]{criticGrad}).getValue();
		critic.accGradParameters();
		
		// Get the actor action for the current state
		Tensor action = actor.forward(batch.getState());
		
		// Get the actor gradient by evaluating the critic and use it's gradient with respect to the action
		// Note: By default we're doing minimization, so set negative critic gradient
		value = critic.forward(inputIds, outputIds, new Tensor[]{batch.getState(), action}).getValue().tensor;
		criticGrad.fill(-1f/config.batchSize);
		Tensor actorGrad = critic.backward(outputIds, inputIds, new Tensor[]{criticGrad}).getValue().tensors.get(actionIn);
		
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
		return new LearnProgress(i, loss, new String[]{"q"}, new float[]{TensorOps.sum(value)/config.batchSize});
	}

}
