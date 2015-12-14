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
package be.iminds.iot.dianne.rl.learn.processors;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.nn.learn.processors.MinibatchProcessor;
import be.iminds.iot.dianne.nn.learn.processors.StochasticGradientDescentProcessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class TimeDifferenceProcessor extends MinibatchProcessor {
	
	private final String[] logLabels = new String[]{"Q", "Target Q", "Error"};
	
	protected final NeuralNetwork target;
	
	protected final ExperiencePool pool;
	
	protected float discountRate = 0.99f;
	
	public TimeDifferenceProcessor(TensorFactory factory,
			NeuralNetwork nn,
			NeuralNetwork target,
			DataLogger logger,
			ExperiencePool pool, 
			SamplingStrategy s,
			Criterion c,
			int batchSize,
			float discount) {
		super(factory, nn, logger, pool, s, c, batchSize);
		
		this.target = target;
		this.pool = pool;
		this.discountRate = discount;
	}

	
	protected Tensor getGradOut(Tensor out, int index){
		
		Tensor action = pool.getAction(index);
		float reward = pool.getReward(index);
		Tensor nextState = pool.getNextState(index);
		
		float targetQ = 0;
		
		if(nextState==null){
			// terminal state
			targetQ = reward;
		} else {
			Tensor nextQ = target.forward(nextState, ""+index);
			targetQ = reward + discountRate * factory.getTensorMath().max(nextQ);
		}
		
		Tensor targetOut = out.copyInto(null);
		targetOut.set(targetQ, factory.getTensorMath().argmax(action));
		
		Tensor e = criterion.error(out, targetOut);
		error += e.get(0);
		
		if(logger!=null){
			logger.log("LEARN", logLabels, factory.getTensorMath().max(out), targetQ, e.get(0));
		}
		
		return criterion.grad(out, targetOut);
		
	}
}
