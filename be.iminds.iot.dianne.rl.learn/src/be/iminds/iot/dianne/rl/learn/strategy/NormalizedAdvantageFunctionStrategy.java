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
package be.iminds.iot.dianne.rl.learn.strategy;

import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.NeuralNetworkResult;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.nn.module.dto.ModuleInstanceDTO;
import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkInstanceDTO;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.learn.QLearnProgress;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.DeepQConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This learning strategy implements NAF
 * 
 * The strategy requires 2 NN instances of the same NN: one acting as a target for the other
 * 
 * In order to make this work, make sure to set the syncInterval of the target to make sure it 
 * updates from time to time to the weights of the trained NN.
 * 
 * @author sbohez
 *
 */
public class NormalizedAdvantageFunctionStrategy implements LearningStrategy {

	protected DeepQConfig config;
	
	protected ExperiencePool pool;
	protected SamplingStrategy sampling;
	
	protected ExperiencePoolBatch batch;
	
	protected NeuralNetwork valueNetwork;
	protected NeuralNetwork targetNetwork;
	
	protected Criterion criterion;
	protected GradientProcessor gradientProcessor;
	
	protected Tensor targetActionValue;
	protected UUID stateIn, actionIn, stateValueOut, actionValueOut;
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset is no experience pool");
		
		this.pool = (ExperiencePool) dataset;
		
		if(nns.length != 2)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 2)");
			
		this.valueNetwork = nns[0];
		this.targetNetwork = nns[1];
		
		this.config = DianneConfigHandler.getConfig(config, DeepQConfig.class);
		this.sampling = SamplingFactory.createSamplingStrategy(this.config.sampling, dataset, config);
		config.put("batchSize", "1");
		this.criterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, valueNetwork, config);
		
		NeuralNetworkInstanceDTO nndto = this.valueNetwork.getNeuralNetworkInstance();
		for(UUID iid : this.valueNetwork.getInputs().keySet()) {
			ModuleInstanceDTO mdto = nndto.modules.get(iid);
			String mname = mdto.module.properties.get("name");
			
			if(mname.equalsIgnoreCase("state"))
				this.stateIn = iid;
			else if(mname.equalsIgnoreCase("action"))
				this.actionIn = iid;
		}
		for(UUID oid : this.valueNetwork.getOutputs().keySet()) {
			ModuleInstanceDTO mdto = nndto.modules.get(oid);
			String mname = mdto.module.properties.get("name");
			
			if(mname.equalsIgnoreCase("statevalue"))
				this.stateValueOut = oid;
			else if(mname.equalsIgnoreCase("actionvalue"))
				this.actionValueOut = oid;
		}
		
		if(stateIn == null || actionIn == null || stateValueOut == null || actionValueOut == null)
			throw new RuntimeException("Unable to select correct Input and Output modules from network " + nndto.name);
		
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
		valueNetwork.zeroDeltaParameters();
		
		// Fill in the batch
		batch = pool.getBatch(batch, sampling.next(config.batchSize));
		
		float totalValue = 0, totalLoss = 0;
		for(int b = 0; b < config.batchSize; b++) {
			// Get the data from the sample
			Tensor state = batch.getState(b);
			Tensor action = batch.getAction(b);
			Tensor reward = batch.getReward(b);
			Tensor nextState = batch.getNextState(b);
			
			// Calculate the target value
			if(!batch.isTerminal(b)) {
				// If the next state is not terminal, get the next value using the target network
				Tensor nextStateValue = targetNetwork.forward(stateIn, stateValueOut, nextState).getValue().tensor;
				
				// Set the target value using the Bellman equation
				targetActionValue = TensorOps.add(targetActionValue, reward, config.discount, nextStateValue);
			} else {
				// If the next state is terminal, the target value is equal to the reward
				targetActionValue = reward.copyInto(targetActionValue);
			}
			
			NeuralNetworkResult result = valueNetwork.forward(new UUID[]{stateIn, actionIn}, new UUID[]{stateValueOut, actionValueOut}, new Tensor[]{state, action}).getValue();
			Tensor stateValue = result.tensors.get(stateValueOut), actionValue = result.tensors.get(actionValueOut);
			
			Tensor loss = criterion.loss(actionValue, targetActionValue);
			Tensor grad = criterion.grad(actionValue, targetActionValue);
			
			valueNetwork.backward(actionValueOut, stateIn, grad, true).getValue();
			
			totalValue += stateValue.get(0);
			totalLoss += loss.get(0);
		}
		
		// Call the processors to set the updates
		gradientProcessor.calculateDelta(i);
		
		// Apply the updates
		// Note: target network gets updated automatically by setting the syncInterval option
		valueNetwork.updateParameters();
		
		return new QLearnProgress(i, totalLoss/config.batchSize, totalValue/config.batchSize);
	}

}
