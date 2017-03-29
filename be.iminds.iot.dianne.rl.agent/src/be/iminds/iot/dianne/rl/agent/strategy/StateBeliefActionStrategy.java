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
	
	private Random rand = new Random();
	
	private StateBeliefConfig config;
	private NeuralNetwork posterior;
	private NeuralNetwork policy;
	
	// prev state (in case we use p(s_t | s_t-1, a_t-1, o_t) )
	private Tensor state;
	// sampled states from posterior
	private Tensor states;

	private Tensor means;
	private Tensor stdevs;
	
	// sampled action
	private Tensor action;
	// histogram for actions based on policy and sampled states
	private Tensor actions;
	
	private UUID[] posteriorIn;
	private UUID[] posteriorOut;
	
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
		
		this.action = new Tensor(env.actionDims());
		this.actions = new Tensor(env.actionDims());
		
		this.states = new Tensor(this.config.batchSize, this.config.stateSize);
		this.means = new Tensor(this.config.batchSize, this.config.stateSize);
		this.stdevs = new Tensor(this.config.batchSize, this.config.stateSize);

		this.state = new Tensor(this.config.stateSize);
	}

	@Override
	public Tensor processIteration(long s, long i, Tensor observation) throws Exception {
		
		if(i == 0){
			// initialize state/action with zero's
			state.fill(0.0f);
			action.fill(0.0f);
		}
		
		// forward o_t, s_t-1, a_t-1 to get state posterior  p(s_t | s_t-1, a_t-1, o_t)
		// TODO we might need to keep all observations and sample p(s_t | a_0..a_t-1, o_0..o_t) here instead? 
		Tensor posteriorParams = posterior.forward(posteriorIn, posteriorOut, new Tensor[]{state, action, observation}).getValue().tensor;

		state = posteriorParams.narrow(0, 0, config.stateSize).copyInto(state);
		
		sampleStates(posteriorParams);
		
		Tensor q = policy.forward(states);

		// determine action from q's
		sampleAction(q);
		
		return action;
	}

	
	private void sampleStates(Tensor posteriorParams){
		// sample states from posterior params
		Tensor mean = posteriorParams.narrow(0, 0, config.stateSize);
		Tensor stdev = posteriorParams.narrow(0, config.stateSize, config.stateSize);
		for(int b = 0;b<config.batchSize;b++){
			mean.copyInto(means.select(0, b));
			stdev.copyInto(stdevs.select(0, b));
		}
		
		states.randn();
		
		TensorOps.cmul(states, states, stdevs);
		TensorOps.add(states, states, means);
	}
	
	private void sampleAction(Tensor q){
		// fill actions tensor
		actions.fill(0.0f);
		for(int b = 0;b<config.batchSize;b++){
			int bestAction = TensorOps.argmax(q.select(0, b));
			actions.set(actions.get(bestAction)+1, bestAction);
		}
		
		// sample action (or get max voted)
		action.fill(0.0f);
		int a = 0;
		if(config.sampleAction){
			int r = rand.nextInt((int)TensorOps.sum(actions));
			float s = 0;
			while (a < actions.size() && (s += actions.get(a)) < r) {
				a++;
			}
		} else {
			a = TensorOps.argmax(actions);
		}
		action.set(1.0f, a);
	}
}
