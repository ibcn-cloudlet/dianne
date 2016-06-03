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

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import be.iminds.iot.dianne.api.log.DataLogger;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.nn.learn.processors.config.RMSpropConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class RMSpropProcessor extends GradientProcessor {

	private final RMSpropConfig config;
	
	private final Map<UUID, Tensor> meanSquared = new HashMap<>();

	private Tensor squared = null;
	
	public RMSpropProcessor( NeuralNetwork nn, DataLogger logger, RMSpropConfig config) {
		super(nn, logger);
		
		this.config = config;
	}
	
	@Override
	public void updateDelta(long i) {
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor deltaParams = e.getValue().getDeltaParameters();
			
			// calculate mean squared
			squared = deltaParams.copyInto(squared);
			squared = TensorOps.cmul(squared, squared, squared);
			
			Tensor mSq = meanSquared.get(e.getKey());
			if(mSq == null){
				mSq = TensorOps.mul(mSq, squared, (1-config.decayRate));
			} else {
				mSq = TensorOps.mul(mSq, mSq, config.decayRate);
				mSq = TensorOps.add(mSq, mSq, (1-config.decayRate), squared);
			}
			meanSquared.put(e.getKey(), mSq);

			// delta params = - learning_rate * dx / np.sqrt(meanSquared + epsiolon)
			TensorOps.mul(deltaParams, deltaParams, -config.learningRate);
			
			// add 1e-8 to avoid div by zero, reuse squared tensor for this
			TensorOps.add(squared, mSq, config.epsilon);
			TensorOps.sqrt(squared, squared);
			
			// now div by cached tensor
			TensorOps.cdiv(deltaParams, deltaParams, squared);
			
			// set DeltaParameters to be sure in case of remote module instance
			e.getValue().setDeltaParameters(deltaParams);
		});
	}

}
