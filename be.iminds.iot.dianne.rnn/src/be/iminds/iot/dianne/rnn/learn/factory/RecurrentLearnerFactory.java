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
package be.iminds.iot.dianne.rnn.learn.factory;

import java.util.Map;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.api.rnn.dataset.SequenceDataset;
import be.iminds.iot.dianne.nn.learn.factory.LearnerFactory;
import be.iminds.iot.dianne.nn.learn.processors.AbstractProcessor;
import be.iminds.iot.dianne.rnn.learn.processors.BpttProcessor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class RecurrentLearnerFactory {

	public static Processor createProcessor(TensorFactory factory, 
			NeuralNetwork nn, 
			SequenceDataset d, 
			Map<String, String> config,
			DataLogger logger){
		AbstractProcessor p = createBpttProcessor(factory, nn, d, config, logger);
		p = LearnerFactory.addRegularization(p, config);
		p = LearnerFactory.addMomentum(p, config);
		return (Processor) p;
	}
	
	public static AbstractProcessor createBpttProcessor(TensorFactory factory, 
			NeuralNetwork nn, SequenceDataset d, Map<String, String> config, DataLogger logger){
		
		int sequenceLength = 20;
		if(config.get("sequenceLength")!=null){
			sequenceLength = Integer.parseInt(config.get("sequenceLength"));
		}
		
		boolean backpropAll = true;
		if(config.containsKey("backpropAll"))
			backpropAll = Boolean.parseBoolean(config.get("backpropAll"));

		Criterion c = LearnerFactory.createCriterion(factory, config);
		SamplingStrategy s = LearnerFactory.createSamplingStrategy(d, config);
		
		System.out.println("Back propagate through time");
		System.out.println("* criterion = "+c.getClass().getName());
		System.out.println("* sampling = "+ s.getClass().getName());
		System.out.println("* sequenceLength = "+sequenceLength);
		System.out.println("* backpropAll = "+backpropAll);
		System.out.println("---");
		
		AbstractProcessor p = new BpttProcessor(factory, nn, logger, d, s, c, 
				 sequenceLength, backpropAll);

		return LearnerFactory.selectSGDMethod(p, config);
	}
}
