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
package be.iminds.iot.dianne.rl.agent.strategy;

import java.util.Map;
import java.util.Random;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.rl.agent.ActionStrategy;
import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.agent.strategy.config.StateBeliefConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Action strategy that uses both a state belief NN and a policy NN 
 * 
 * @author tverbele
 *
 */
public class StateBeliefActionStrategy implements ActionStrategy {
	
	private StateBeliefConfig config;
	private NeuralNetwork posterior;
	private NeuralNetwork policy;

	private NeuralNetwork prior;
	private float drop = 0.0f;
	private int warmup = 0;
	
	// prev state samples (in case we use p(s_t | s_t-1, a_t-1, o_t) )
	private Tensor state;
	
	// state parameters
	private Tensor params;
	
	// action options 
	private Tensor action;
	
	// observation expanded
	private Tensor observation;
	
	// q's
	private Tensor q;
	
	// selected action
	private Tensor act;
	
	private UUID[] posteriorIn;
	private UUID[] posteriorOut;
	
	private UUID[] priorIn;
	private UUID[] priorOut;
	
	private Random random = new Random();

	
	@Override
	public void setup(Map<String, String> config, Environment env, NeuralNetwork... nns) throws Exception {
		this.config = DianneConfigHandler.getConfig(config, StateBeliefConfig.class);
		if(nns.length < 2){
			throw new RuntimeException("Provide both a state belief and policy neural network for this strategy!");
		}
		// TODO check whether the #inputs are correct?
		this.posterior = nns[0];
		this.posteriorIn = posterior.getModuleIds("State","Action","Observation");
		this.posteriorOut = new UUID[]{posterior.getOutput().getId()};
		this.policy = nns[1];
		
		if(nns.length >= 3) {
			this.prior = nns[2];
			this.priorIn = prior.getModuleIds("State","Action");
			this.priorOut = new UUID[]{prior.getOutput().getId()};
			
			if(config.containsKey("drop")) {
				this.drop = Float.parseFloat(config.get("drop"));
			}
			
			if(config.containsKey("warmup")) {
				this.warmup = Integer.parseInt(config.get("warmup"));
			}
		}
		
		this.act = new Tensor(env.actionDims());
		
		// separate case for noSamples = 1, else potentially problematic reshapes in NNs
		this.state = this.config.noSamples > 1 ? new Tensor(this.config.noSamples, this.config.stateSize) : new Tensor(this.config.stateSize);
		this.observation = this.config.noSamples > 1 ? new Tensor(this.config.noSamples, env.observationDims()) : new Tensor(env.observationDims());
		this.action = this.config.noSamples > 1 ? new Tensor(this.config.noSamples, env.actionDims()) : new Tensor(env.actionDims());
	}

	@Override
	public Tensor processIteration(long s, long i, Tensor obs) throws Exception {
		
		if(i == 0){
			// initialize state/action with zero's
			state.fill(0.0f);
			action.fill(0.0f);
		}

		// allow for combination with epsilon greedy exploration
		double epsilon = config.epsilonMin + (config.epsilonMax - config.epsilonMin) * Math.exp(-s * config.epsilonDecay);
		
		if (Math.random() < epsilon) {
			if(config.momentum > 0.0f){
				if(Math.random() < config.momentum){
					return act;
				}
			} 
			act.fill(0);
			act.set(1, (int) (Math.random() * act.size()));
			return act;
		} else {
		
			if(drop > 0 && i >= warmup && Math.random() < drop) {
				// sample from prior
				params = prior.forward(priorIn, priorOut, new Tensor[] {state, action}).getValue().tensor;
			} else {
				// forward o_t, s_t-1, a_t-1 to get state posterior  p(s_t | s_t-1, a_t-1, o_t)
				if(config.noSamples > 1)
					observation = TensorOps.expand(observation, obs, config.noSamples);
				else 
					observation = obs;
				
				params = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{state, action, observation}).getValue().tensor;
			}
			
			if(config.noSamples > 1) 
				state = sampleFromGaussianMixture(state, params, config.noSamples);
			else
				state = sampleFromGaussian(state, params);
			
			q = policy.forward(state);
			
			int best = 0;
			
			if(config.noSamples > 1) {
				// use majority voting to select actual action?!
				act.fill(0);
				for(int k=0;k<config.noSamples;k++) {
					int a = TensorOps.argmax(q.select(0, k));
					act.set(act.get(a)+1, a);
				}
				
				best = TensorOps.argmax(act);
			} else {
				best = TensorOps.argmax(q);
			}
			
			act.fill(0);
			act.set(1, best);
			
			if(config.noSamples > 1)
				TensorOps.expand(action, act, config.noSamples);
			else 
				action = act;
			
			return act;
		}
	}
	
	private Tensor sampleFromGaussianMixture(Tensor result, Tensor distribution, int batchSize) {
		if(batchSize > 1) {
			for(int i = 0; i < batchSize; i++)
				sampleFromGaussianMixture(result.select(0, i), distribution);
		}
		return result;
	}
	
	private Tensor sampleFromGaussianMixture(Tensor result, Tensor distribution) {
		return sampleFromGaussian(result, distribution.select(0, random.nextInt(distribution.size(0))));
	}
	
	private Tensor sampleFromGaussian(Tensor result, Tensor distribution) {
		int size = distribution.size()/2;
		Tensor means = distribution.narrow(0, 0, size);
		Tensor stdevs = distribution.narrow(0, size, size);
		
		Tensor random = new Tensor(means.size());
		random.randn();
		
		TensorOps.cmul(result, random, stdevs);
		TensorOps.add(result, result, means);
		return result;
	}

}
