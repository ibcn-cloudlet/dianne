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

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.rl.agent.ActionController;
import be.iminds.iot.dianne.api.rl.agent.ActionStrategy;
import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.tensor.Tensor;

public class ManualActionStrategy implements ActionStrategy, ActionController {

	// in case you want to wait in each state for a new setAction
	private boolean wait = false;
	
	private Tensor action = null;
	
	public void setAction(Tensor a){
		synchronized(this){
			this.action = a;
			this.notifyAll();
		}
	}

	@Override
	public void setup(Map<String, String> config, Environment env, NeuralNetwork... nns) throws Exception {
		if(config.containsKey("wait")){
			wait = Boolean.parseBoolean(config.get("wait"));
		}
	}

	@Override
	public Tensor processIteration(long s, long i, Tensor state) throws Exception {
		synchronized(this){
			if(wait || action == null){
				try {
					this.wait();
				} catch(InterruptedException e){}
			}
			return action;
		}
	}
	
}
