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

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.rl.agent.ActionStrategy;
import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;
import be.iminds.iot.dianne.rl.agent.strategy.config.RandomConfig;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Select a random discrete action.
 * 
 * @author tverbele
 *
 */
public class RandomActionStrategy implements ActionStrategy {

	private RandomConfig config;
	private Random random = new Random(System.currentTimeMillis());
	
	private Tensor action;
	
	@Override
	public void setup(Map<String, String> config, Environment env, NeuralNetwork... nns) throws Exception {
		this.config = DianneConfigHandler.getConfig(config, RandomConfig.class);
		this.action = new Tensor(env.actionDims());
	}

	@Override
	public Tensor processIteration(long s, long i, Tensor state) throws Exception {
		
		if(config.discrete){
			int a = random.nextInt(action.size());
			action.fill(0.0f);
			action.set(1.0f, a);
		} else {
			action.randn();
			ModuleOps.tanh(action, action);
		}
		
		return action;
	}
	
}
