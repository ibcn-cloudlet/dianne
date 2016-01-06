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
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class StochasticGradientDescentProcessor extends GradientProcessor {

	private final float learningRate;
	
	public StochasticGradientDescentProcessor( TensorFactory factory, NeuralNetwork nn, DataLogger logger, float learningRate ) {
		super(factory, nn, logger);
		this.learningRate = learningRate;
	}
	
	@Override
	public void updateDelta(long i) {
		// apply learning rate
		nn.getTrainables().values().stream().forEach(m -> {
			Tensor deltaParams = m.getDeltaParameters();

			factory.getTensorMath().mul(deltaParams, deltaParams, -learningRate);
					
			// set DeltaParameters to be sure in case of remote module instance
			m.setDeltaParameters(deltaParams);
		});
	}

}
