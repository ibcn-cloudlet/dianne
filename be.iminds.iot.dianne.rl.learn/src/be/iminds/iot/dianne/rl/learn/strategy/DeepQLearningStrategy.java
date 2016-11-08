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

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolBatch;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.learn.QLearnProgress;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.learn.sampling.SamplingFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.DeepQConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This learning strategy implements DQN
 * 
 * The strategy requires 2 NN instances of the same NN: one acting as a target for the other
 * 
 * In order to make this work, make sure to set the syncInterval of the target to make sure it 
 * updates from time to time to the weights of the trained NN.
 * 
 * @author tverbele
 *
 */
public class DeepQLearningStrategy implements LearningStrategy {

	protected DeepQConfig config;
	
	protected ExperiencePool pool;
	protected SamplingStrategy sampling;
	protected ExperiencePoolBatch batch;
	
	protected NeuralNetwork valueNetwork;
	protected NeuralNetwork targetNetwork;
	
	protected Criterion criterion;
	protected GradientProcessor gradientProcessor;
	
	protected Tensor actionBatch;
	protected Tensor targetValueBatch;
	
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
		this.criterion = CriterionFactory.createCriterion(this.config.criterion, config);
		this.gradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, valueNetwork, config);
		
		// Pre-allocate tensors for batch operations
		this.actionBatch = new Tensor(this.config.batchSize, this.pool.actionDims()[0]);
		this.targetValueBatch = new Tensor(this.config.batchSize, this.pool.actionDims()[0]);
		
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
		
		// Reset the action & target value batch
		// Note: actionBatch is a reverse mask of the action selected
		actionBatch.fill(1);
		targetValueBatch.fill(0);
		
		// Fill in the batch
		batch = pool.getBatch(batch, sampling.next(config.batchSize));
		
		for(int b = 0; b < config.batchSize; b++) {
			// Get the data from the sample
			// Note: actions are one-hot encoded
			int action = TensorOps.argmax(batch.getAction(b));
			Tensor nextState = batch.getNextState(b);
			float reward = batch.getScalarReward(b);
			
			// Flag proper action
			actionBatch.set(0, b, action);
			
			// Calculate the target value
			if(!batch.isTerminal(b)) {
				// If the next state is not terminal, get the next value using the target network
				Tensor nextValue = targetNetwork.forward(nextState);
				
				// Determine the next action, depends on whether we are using double Q learning or not
				int nextAction = TensorOps.argmax(config.doubleQ ? valueNetwork.forward(nextState) : nextValue);
				
				// Set the target value using the Bellman equation
				targetValueBatch.set(reward + config.discount*nextValue.get(nextAction), b, action);
			} else {
				// If the next state is terminal, the target value is equal to the reward
				targetValueBatch.set(reward, b, action);
			}
		}
		
		// Forward pass of the value network to get the current value estimate
		Tensor valueBatch = valueNetwork.forward(batch.getState());
		
		// Fill in the missing target values
		TensorOps.addcmul(targetValueBatch, targetValueBatch, 1, actionBatch, valueBatch);
		
		// Get the total value for logging and calculate the MSE error and gradient with respect to the target value
		float value = TensorOps.sum(valueBatch)/config.batchSize/pool.actionDims()[0];
		float loss = criterion.loss(valueBatch, targetValueBatch);
		Tensor grad = criterion.grad(valueBatch, targetValueBatch);
		
		// Backward pass of the critic
		valueNetwork.backward(grad);
		valueNetwork.accGradParameters();
		
		// Call the processos to set the updates
		gradientProcessor.calculateDelta(i);
		
		// Apply the updates
		// Note: target network gets updated automatically by setting the syncInterval option
		valueNetwork.updateParameters();
		
		return new QLearnProgress(i, loss, value);
	}

}
