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

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.nn.learn.processors.config.SGDConfig;
import be.iminds.iot.dianne.nn.learn.processors.config.SGDConfig.DecayType;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class StochasticGradientDescentProcessor extends GradientProcessor {

	private final SGDConfig config;
	
	public StochasticGradientDescentProcessor(NeuralNetwork nn, DataLogger logger, SGDConfig config) {
		super(nn, logger);
		this.config = config;
	}
	
	@Override
	public void updateDelta(long i) {
		float learningRate = config.learningRate;
		if(config.decayRate > 0){
			if(config.decayType == DecayType.EXPONENTIAL){
				learningRate = (float) (config.minLearningRate + (config.learningRate - config.minLearningRate)*Math.exp(-i * config.decayRate));
			} else if(config.decayType == DecayType.LINEAR){
				learningRate = config.learningRate - config.decayRate*i;
				if(learningRate < config.minLearningRate){
					learningRate = config.minLearningRate;
				}
			}
			if(config.trace){
				System.out.println("Learning rate: "+learningRate);
			}
		}
		
		final float rate = learningRate;
		nn.getTrainables().values().stream().forEach(m -> {
			// Get the gradients
			Tensor deltaParams = m.getDeltaParameters();

			// Apply learning rate
			TensorOps.mul(deltaParams, deltaParams, -rate);
					
			// Set DeltaParameters to be sure in case of remote module instance
			m.setDeltaParameters(deltaParams);
		});
	}
}
