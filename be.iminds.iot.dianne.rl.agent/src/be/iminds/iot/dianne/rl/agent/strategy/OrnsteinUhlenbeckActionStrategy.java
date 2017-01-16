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
package be.iminds.iot.dianne.rl.agent.strategy;

import java.util.Map;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.rl.agent.ActionStrategy;
import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.agent.strategy.config.OrnsteinUhlenbeckConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * In the literature, they use the Ornstein-Uhlenbeck stochastic
 * process for control tasks that deal with momentum
 * 
 * @author edconinc
 *
 */
public class OrnsteinUhlenbeckActionStrategy implements ActionStrategy {
	
	private NeuralNetwork policy;
	private OrnsteinUhlenbeckConfig config;
	
	private Tensor noise;
	private Tensor whiteNoise;

	@Override
	public void setup(Map<String, String> config, Environment env, NeuralNetwork... nns) throws Exception {
		this.policy = nns[0];
		this.config = DianneConfigHandler.getConfig(config, OrnsteinUhlenbeckConfig.class);
		this.noise = new Tensor(env.actionDims());
		this.noise.fill(0f);
		this.whiteNoise = new Tensor(env.actionDims());
	}

	@Override
	public Tensor processIteration(long i, Tensor state) throws Exception {
		Tensor action = policy.forward(state);
		
		if (i % config.maxActions == 0) {
			noise.fill(0f);
		}
		if (config.explorationActions != -1 && i % config.maxActions > config.explorationActions) {
			return action;
		}
		whiteNoise.randn();
		
		// TODO: different config parameters for each action
		for (int j=0; j<action.size(); j++) { // use previous noise for each action
			// Solve using Euler-Maruyama method
			float val = config.theta * (config.mu - action.get(j)) + config.sigma * whiteNoise.get()[j];
			noise.set(val, j);
		}
		double stdev = config.noiseMin + (config.noiseMax - config.noiseMin) * Math.exp(-i * config.noiseDecay);
				
		TensorOps.add(action, action, (float) stdev, noise);
		
		for(int a = 0; a < action.size(); a++) {
			float v = action.get(a);
			if(v < config.minValue)
				action.set(config.minValue, a);
			else if(v > config.maxValue)
				action.set(config.maxValue, a);
		}
		return action;
	}

}
