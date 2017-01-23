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
import be.iminds.iot.dianne.api.rl.agent.ActionStrategy;
import be.iminds.iot.dianne.api.rl.environment.Environment;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Discrete sampling ActionStrategy that takes the output of a (discrete) policy network
 * ending with a softmax/logsoftmax module and samples an actual discrete action from it.  
 * @author tverbele
 *
 */
public class DiscreteSamplingActionStrategy implements ActionStrategy {
	
	private NeuralNetwork nn;
	
	@Override
	public void setup(Map<String, String> config, Environment env, NeuralNetwork... nns) throws Exception {
		this.nn = nns[0];
	}

	@Override
	public Tensor processIteration(long s, long i, Tensor state) throws Exception {
		Tensor output = nn.forward(state);
		
		Tensor action = new Tensor(output.size());
		action.fill(0);
		
		if(TensorOps.min(output) < 0){
			// assume logsoftmax output, take exp
			output = TensorOps.exp(output, output);
		}
		
		double t = 0, r = Math.random();
		int a = 0;
		while(a < output.size() && (t += output.get(a)) < r){
			a++;
		}
		
		action.set(1, a);
		
		return action;
	}

}
