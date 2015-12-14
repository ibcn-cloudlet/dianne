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
package be.iminds.iot.dianne.rl.learn.factory;

import java.util.Map;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rl.ExperiencePool;
import be.iminds.iot.dianne.nn.learn.factory.LearnerFactory;
import be.iminds.iot.dianne.nn.learn.processors.AbstractProcessor;
import be.iminds.iot.dianne.nn.learn.processors.StochasticGradientDescentProcessor;
import be.iminds.iot.dianne.rl.learn.processors.TimeDifferenceProcessor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class QLearnerFactory {

	public static Processor createProcessor(
		TensorFactory factory, 
		NeuralNetwork nn, 
		NeuralNetwork target,
		ExperiencePool pool, 
		Map<String, String> config,
		DataLogger logger){

		AbstractProcessor p = createTDProcessor(factory, nn, target, pool, config, logger);
		p = LearnerFactory.addRegularization(p, config);
		p = LearnerFactory.addMomentum(p, config);
		return (Processor) p;
	}
	
	public static AbstractProcessor createTDProcessor(TensorFactory factory, 
			NeuralNetwork nn, NeuralNetwork target,
			ExperiencePool pool, Map<String, String> config, DataLogger logger){
		
		float learningRate = 0.001f;
		if(config.get("learningRate")!=null){
			learningRate = Float.parseFloat(config.get("learningRate"));
		}
		
		int batchSize = 10;
		if(config.get("batchSize")!=null){
			batchSize = Integer.parseInt(config.get("batchSize"));
		}
		
		float discount = 0.99f;
		if(config.containsKey("discount"))
			discount = Float.parseFloat(config.get("discount"));
		
		Criterion c = LearnerFactory.createCriterion(factory, config);
		SamplingStrategy s = LearnerFactory.createSamplingStrategy(pool, config);
		
		AbstractProcessor p = new StochasticGradientDescentProcessor(new TimeDifferenceProcessor(factory, nn, target, logger, pool, s, c, 
				 batchSize, discount), learningRate);
		
		System.out.println("TimeDifferenceRL");
		System.out.println("* criterion = "+c.getClass().getName());
		System.out.println("* sampling = "+ s.getClass().getName());
		System.out.println("* learningRate = "+learningRate);
		System.out.println("* batchSize = "+batchSize);
		System.out.println("* discount = "+discount);
		System.out.println("---");
		
		return p;
	}
}
