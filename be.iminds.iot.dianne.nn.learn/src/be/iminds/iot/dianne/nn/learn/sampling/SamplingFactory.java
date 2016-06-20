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
package be.iminds.iot.dianne.nn.learn.sampling;

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
import be.iminds.iot.dianne.nn.learn.processors.AdadeltaProcessor;
import be.iminds.iot.dianne.nn.learn.processors.AdagradProcessor;
import be.iminds.iot.dianne.nn.learn.processors.MomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.NesterovMomentumProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RMSpropProcessor;
import be.iminds.iot.dianne.nn.learn.processors.RegularizationProcessor;
import be.iminds.iot.dianne.nn.learn.processors.StochasticGradientDescentProcessor;
import be.iminds.iot.dianne.nn.learn.processors.config.AdadeltaConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.AdagradConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.MomentumConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.NesterovConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.RMSpropConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.RegularizationConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.SGDConfig;
import be.iminds.iot.dianne.nn.util.DianneConfigHandler;

public class SamplingFactory {
	
	public static SamplingStrategy createSamplingStrategy(LearnerConfig.Sampling strategy, Dataset d, Map<String, String> config){
		SamplingStrategy sampling = null;

		switch(strategy) {
		case SEQUENTIAL:
			sampling = new SequentialSamplingStrategy(d);
			break;
		default:
			sampling = new RandomSamplingStrategy(d);
		}
		
		return sampling;
	}
}
