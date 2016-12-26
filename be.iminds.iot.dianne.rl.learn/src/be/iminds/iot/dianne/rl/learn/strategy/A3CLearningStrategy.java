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
import java.util.Random;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sequence;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.LearnProgress;
import be.iminds.iot.dianne.api.nn.learn.LearningStrategy;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePool;
import be.iminds.iot.dianne.api.rl.dataset.ExperiencePoolSample;
import be.iminds.iot.dianne.api.rl.learn.A3CLearnProgress;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory;
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
	protected Criterion valueCriterion;
	
	protected A3CConfig config;
	
	protected Random r = new Random(System.currentTimeMillis());
	
	protected Tensor policyGrad = null;
	
	protected Tensor probs = null;
	protected Tensor entropyGrad = null;
	
	protected Tensor actorGrad = null;
	protected Tensor criticGrad = null;
	
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
		this.valueCriterion = CriterionFactory.createCriterion(this.config.valueCriterion, config);
		this.policyGradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, policyNetwork, config);
		this.valueGradientProcessor = ProcessorFactory.createGradientProcessor(this.config.method, valueNetwork, config);
	}

	@Override
	public LearnProgress processIteration(long i) throws Exception {
		// Reset the deltas
		policyNetwork.zeroDeltaParameters();
		valueNetwork.zeroDeltaParameters();


		// Wait for the pool to contain sequences
		if(pool.sequences() == 0){
			while(pool.sequences() == 0 ){
				try {
					Thread.sleep(5000);
				} catch (InterruptedException e) {
				}
			}
		}
		
		// TODO reuse samples for sequence?
		// how to determine the correct length then in case a too long sequence is provided?
		
		// either get a random sequence or get and remove the first sequence from the pool
		Sequence<ExperiencePoolSample> sequence = null;
		if(config.reuseSequences){
			int index = r.nextInt(pool.sequences());
			sequence = pool.getSequence(index);
		} else {
			sequence = pool.removeAndGetSequence(0);
		}	
		
		float reward = 0;
		ExperiencePoolSample last = sequence.get(sequence.size()-1);
		if(!last.isTerminal()){
			reward = valueNetwork.forward(last.getNextState()).get(0);
		}
		
		float loss = 0;
		float total_value = 0;
		float entropy = 0;
		float total_entropy = 0;
		
		for(int k=sequence.size()-1;k>=0;k--){
			ExperiencePoolSample sample = sequence.get(k);
			
			// update expected discounted reward
			reward = config.discount * reward + sample.getScalarReward();

			// calculate value of state
			Tensor value = valueNetwork.forward(sample.getState());
			total_value+= value.get(0);
			
			// also update policy
			if(config.updatePolicy){
				// calculate action log probabilities from policy network
			
				Tensor log_probs = policyNetwork.forward(sample.getState());
				
				// calculate policy gradient
				if(policyGrad == null)
					policyGrad  = new Tensor(log_probs.size());
				
				policyGrad.fill(0.0f);
				
				int action = TensorOps.argmax(sample.getAction());
				
				float advantage = reward - value.get(0);
				policyGrad.set(-advantage, action);
				
				actorGrad = policyGrad.copyInto(actorGrad);
				
				// calculate entropy
				// assume output is logprob
				if(probs == null)
					probs = new Tensor(log_probs.size());
				
				probs = TensorOps.exp(probs, log_probs);
				entropy = -TensorOps.dot(probs, log_probs);
				total_entropy += entropy;
				
				if(config.entropy > 0){
					// entropy regularization
					if(entropyGrad==null)
						entropyGrad = new Tensor(log_probs.size());
					// entropy = - sum_i  p_i * log p_i
					// however, we assume the output is logsoftmax'ed , so gradient to log_pi needed
					// hence, gradient to log_pi becomes :  -(p_i + log_p_i * pi_i)
					entropyGrad = TensorOps.cmul(entropyGrad, probs, log_probs);
					entropyGrad = TensorOps.add(entropyGrad, entropyGrad, probs);
					
					// for exploration one wants a slight push towards higher entropy
					TensorOps.add(actorGrad, policyGrad, config.entropy, entropyGrad);
				}
						
				policyNetwork.backward(actorGrad);
				policyNetwork.accGradParameters();
			}
			
			
			// calculate value gradient
			Tensor target = new Tensor(1);
			target.set(reward, 0);
			
			loss += TensorOps.mean(valueCriterion.loss(value, target));
			
			criticGrad = valueCriterion.grad(value, target);
			
			valueNetwork.backward(criticGrad);
			valueNetwork.accGradParameters();
		}
		
		valueGradientProcessor.calculateDelta(i);
		valueNetwork.updateParameters();

		if(config.updatePolicy){
			policyGradientProcessor.calculateDelta(i);
			policyNetwork.updateParameters();
		}
		
		return new A3CLearnProgress(i, loss/sequence.size(), total_value/sequence.size(), total_entropy/sequence.size());
	}

}
