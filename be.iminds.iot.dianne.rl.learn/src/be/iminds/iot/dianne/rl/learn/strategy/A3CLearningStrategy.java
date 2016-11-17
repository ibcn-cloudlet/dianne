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

import java.util.List;
import java.util.Map;
import java.util.Random;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.learn.QLearnProgress;
import be.iminds.iot.dianne.nn.learn.processors.ProcessorFactory;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.learn.strategy.config.A3CConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This learning strategy implements a variant of A3C: asynchronous advantage actor-critic
 *  
 * The strategy requires 2 NN instances: a policy network and a value network.
 * The policy network should end with a LogSoftmax module.
 * The value network should end with a single scalar value
 * 
 * In order to make this work, use a small cycling experience pool that is constantly filled 
 * with new sequences and a small syncInterval. The original A3C works on-policy, but this way 
 * we can keep the difference between the agent and learner policy small.
 * 
 * @author tverbele
 *
 */
public class A3CLearningStrategy implements LearningStrategy {

	protected ExperiencePool pool;
	
	protected NeuralNetwork policyNetwork;
	protected NeuralNetwork valueNetwork;
	
	protected GradientProcessor policyGradientProcessor;
	protected GradientProcessor valueGradientProcessor;
	
	protected A3CConfig config;
	
	protected Random r = new Random(System.currentTimeMillis());
	
	@Override
	public void setup(Map<String, String> config, Dataset dataset, NeuralNetwork... nns) throws Exception {
		if(!(dataset instanceof ExperiencePool))
			throw new RuntimeException("Dataset is no experience pool");
		
		this.pool = (ExperiencePool) dataset;
		
		if(nns.length != 2)
			throw new RuntimeException("Invalid number of NN instances provided: "+nns.length+" (expected 2)");
			
		this.policyNetwork = nns[0];
		this.valueNetwork = nns[1];
		
		this.config = DianneConfigHandler.getConfig(config, A3CConfig.class);
		this.policyGradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, policyNetwork, config);
		this.valueGradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, valueNetwork, config);

		
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
		policyNetwork.zeroDeltaParameters();
		valueNetwork.zeroDeltaParameters();
		
		int s = r.nextInt(pool.sequences());
		// TODO reuse samples for sequence?
		// how to determine the correct length then in case a too long sequence is proviced?
		List<ExperiencePoolSample> sequence = pool.getSequence(s);
			
		float reward = 0;
		ExperiencePoolSample last = sequence.get(sequence.size()-1); 
		if(!last.isTerminal()){
			reward = valueNetwork.forward(last.getNextState()).get(0);
		}
		
		float loss = 0;
		float value = 0;
		for(int k=sequence.size()-1;k>=0;k--){
			ExperiencePoolSample sample = sequence.get(k);
			
			// update expected discounted reward
			reward += config.discount * sample.getScalarReward();
			
			// calculate action log probabilities from policy network
			Tensor policy = policyNetwork.forward(sample.getState());
			
			// calculate value of state
			value = valueNetwork.forward(sample.getState()).get(0);
			
			// calculate policy gradient
			Tensor policyGrad  = new Tensor(policy.size());
			policyGrad.fill(0.0f);
			
			int action = TensorOps.argmax(sample.getAction());
			
			float diff = reward - value;	
			policyGrad.set(diff, action);
			
			policyNetwork.backward(policyGrad);
			policyNetwork.accGradParameters();
			
			// calculate value gradient
			loss += diff*diff;
			Tensor valueGrad = new Tensor(1);
			valueGrad.set(-2*diff, 0);
			
			valueNetwork.backward(valueGrad);
			valueNetwork.accGradParameters();
			
		}
		
		policyGradientProcessor.calculateDelta(i);
		valueGradientProcessor.calculateDelta(i);
		
		policyNetwork.updateParameters();
		valueNetwork.updateParameters();
		
		return new QLearnProgress(i, loss/sequence.size(), value);
	}

}
