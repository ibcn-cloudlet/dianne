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
	private Tensor deltaNoise;
	
	private Tensor mu;
	private Tensor sigma;
	private Tensor theta;
	private Tensor theta_mu;

	@Override
	public void setup(Map<String, String> config, Environment env, NeuralNetwork... nns) throws Exception {
		this.policy = nns[0];
		this.config = DianneConfigHandler.getConfig(config, OrnsteinUhlenbeckConfig.class);
		
		this.noise = new Tensor(env.actionDims());
		this.deltaNoise = new Tensor(env.actionDims());
		
		String warning = "";
		if (this.config.mu.length == this.noise.size()) {
			this.mu = new Tensor(this.config.mu, env.actionDims());
		} else {
			this.mu = new Tensor(env.actionDims());
			this.mu.fill(this.config.mu[0]);
			warning += "mu, ";
		}
		
		if (this.config.sigma.length == this.noise.size()) {
			this.sigma = new Tensor(this.config.sigma, env.actionDims());
		} else {
			this.sigma = new Tensor(env.actionDims());
			this.sigma.fill(this.config.sigma[0]);
			warning += "sigma, ";
		}
		
		if (this.config.theta.length == this.noise.size()) {
			this.theta = new Tensor(this.config.theta, env.actionDims());
		} else {
			this.theta = new Tensor(env.actionDims());
			this.theta.fill(this.config.theta[0]);
			warning += "theta, ";
		}
		
		if (!warning.isEmpty()) {
			System.err.println(OrnsteinUhlenbeckActionStrategy.class.getName() + ": the config variables [" + warning + "] mismatched with the action space size. Using the first value for all actions.");
		}
		
		this.theta_mu = TensorOps.cmul(this.theta_mu, this.theta, this.mu);
	}

	@Override
	public Tensor processIteration(long s, long i, Tensor state) throws Exception {
		Tensor action = policy.forward(state);
		
		if(i == 0)
			mu.copyInto(noise);

		// noise += ( theta *  mu - theta * noise + sigma * N(0,1) )
		// Solve using Euler-Maruyama method
		deltaNoise.randn();
		TensorOps.cmul(deltaNoise, deltaNoise, sigma);
		TensorOps.addcmul(deltaNoise, deltaNoise, -1, theta, noise);
		TensorOps.add(deltaNoise, deltaNoise, theta_mu);
		TensorOps.add(noise, noise, deltaNoise);
		
		// Decay epsilon
		double eps = config.noiseMin + (config.noiseMax - config.noiseMin) * Math.exp(-s * config.noiseDecay);
		TensorOps.add(action, action, (float) eps, noise);
		
		// clip action [config.minValue, config.maxValue]
		return TensorOps.clamp(action, action, config.minValue, config.maxValue);
	}

}
