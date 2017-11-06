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
	
	// prev state (in case we use p(s_t | s_t-1, a_t-1, o_t) )
	private Tensor state;
	
	private Tensor params;
	
	// sampled action
	private Tensor action;
	
	private UUID[] posteriorIn;
	private UUID[] posteriorOut;
	
	private UUID[] priorIn;
	private UUID[] priorOut;
	
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
		
		this.action = new Tensor(env.actionDims());
		
		this.params = new Tensor(2*this.config.stateSize);

		this.state = new Tensor(this.config.stateSize);
	}

	@Override
	public Tensor processIteration(long s, long i, Tensor observation) throws Exception {
		
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
					return action;
				}
			} 
			action.fill(0);
			action.set(1, (int) (Math.random() * action.size()));
			return action;
		} else {
		
			if(drop > 0 && i >= warmup && Math.random() < drop) {
				// sample from prior
				params = prior.forward(priorIn, priorOut, new Tensor[] {state, action}).getValue().tensor;
			} else {
				// forward o_t, s_t-1, a_t-1 to get state posterior  p(s_t | s_t-1, a_t-1, o_t)
				// TODO we might need to keep all observations and sample p(s_t | a_0..a_t-1, o_0..o_t) here instead? 
				params = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{state, action, observation}).getValue().tensor;
			}
			
			// get an action by sampling a state and determining the Q values
			// this is equivalent with Thompson Sampling https://en.wikipedia.org/wiki/Thompson_sampling
			sampleState();
			
			Tensor q = policy.forward(state);
			action.fill(0);
			action.set(1, TensorOps.argmax(q));
			return action;
		}
	}

	
	private void sampleState(){
		// sample states from params
		Tensor mean = params.narrow(0, 0, config.stateSize);
		Tensor stdev = params.narrow(0, config.stateSize, config.stateSize);
		
		state.randn();
		
		TensorOps.cmul(state, state, stdev);
		TensorOps.add(state, state, mean);
	}
	
}
