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
package be.iminds.iot.dianne.nn.learn.processors;

import java.util.Map;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.api.nn.learn.SamplingStrategy;
import be.iminds.iot.dianne.nn.learn.config.LearnerConfig;
import be.iminds.iot.dianne.nn.learn.criterion.AbsCriterion;
import be.iminds.iot.dianne.nn.learn.criterion.MSECriterion;
import be.iminds.iot.dianne.nn.learn.criterion.NLLCriterion;
import be.iminds.iot.dianne.nn.learn.processors.config.AdadeltaConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.AdagradConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.MomentumConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.NesterovConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.RMSpropConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.RegularizationConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.SGDConfig;
import be.iminds.iot.dianne.nn.learn.sampling.RandomSamplingStrategy;
import be.iminds.iot.dianne.nn.learn.sampling.SequentialSamplingStrategy;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;

public class ProcessorFactory {

	public static GradientProcessor createGradientProcessor(LearnerConfig.Method method, NeuralNetwork nn,
			Map<String, String> config,DataLogger logger){
		return addMomentum(addRegularization(createSGDProcessor(method, nn, config, logger), config), config);
	}
	
	public static GradientProcessor createSGDProcessor(LearnerConfig.Method method, NeuralNetwork nn,
			Map<String, String> config, DataLogger logger){
		
		GradientProcessor p = null;
		
		switch(method) {
		case ADADELTA:
			p = new AdadeltaProcessor(nn, logger, DianneConfigHandler.getConfig(config, AdadeltaConfig.class));
			break;
		case ADAGRAD:
			p = new AdagradProcessor(nn, logger, DianneConfigHandler.getConfig(config, AdagradConfig.class));
			break;
		case RMSPROP:
			p = new RMSpropProcessor(nn, logger, DianneConfigHandler.getConfig(config, RMSpropConfig.class));
			break;	
		default:
			p = new StochasticGradientDescentProcessor(nn, logger, DianneConfigHandler.getConfig(config, SGDConfig.class));
			break;
		}

		return p;
	}
	
	public static GradientProcessor addRegularization(GradientProcessor p, Map<String, String> config){
		if(config.containsKey("l2")) {
			RegularizationProcessor r = new RegularizationProcessor(p, DianneConfigHandler.getConfig(config, RegularizationConfig.class));
			return r;
		} else {
			return p;
		}
	}
	
	public static GradientProcessor addMomentum(GradientProcessor p, Map<String, String> config){
		if(config.containsKey("momentum")) {
			GradientProcessor m = new MomentumProcessor(p, DianneConfigHandler.getConfig(config, MomentumConfig.class));
			return m;
		} else if(config.containsKey("nesterov")) {
			GradientProcessor m = new NesterovMomentumProcessor(p, DianneConfigHandler.getConfig(config, NesterovConfig.class));
			return m;
		} else {
			return p;
		}
	}
	
}
