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
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class AdadeltaProcessor extends GradientProcessor {

	private final float decayRate;
	
	private final Map<UUID, Tensor> meanSquaredGradient = new HashMap<>();
	private final Map<UUID, Tensor> meanSquaredDelta = new HashMap<>();

	private Tensor squared = null;
	
	public AdadeltaProcessor(NeuralNetwork nn, DataLogger logger, float decayRate) {
		super(nn, logger);
		
		this.decayRate = decayRate;
	}
	
	@Override
	public void updateDelta(long i) {
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor deltaParams = e.getValue().getDeltaParameters();
			
			// calculate mean squared gradient
			squared = deltaParams.copyInto(squared);
			squared = TensorOps.cmul(squared, squared, squared);
			
			Tensor mSq = meanSquaredGradient.get(e.getKey());
			if(mSq == null){
				mSq = TensorOps.mul(mSq, squared, (1-decayRate));
			} else {
				mSq = TensorOps.mul(mSq, mSq, decayRate);
				mSq = TensorOps.add(mSq, mSq, (1-decayRate), squared);
			}
			meanSquaredGradient.put(e.getKey(), mSq);
			
			// calculate delta update
			Tensor deltaSq = meanSquaredDelta.get(e.getKey());
			if(deltaSq==null){
				deltaSq = new Tensor(deltaParams.dims());
				deltaSq.fill((float)1e-8);
				meanSquaredDelta.put(e.getKey(), deltaSq);
			} 
			
			// divide mean squared delta by mean squared gradient + 1e-8
			// update = - RMS(delta)/RMS(grad) * grad
			TensorOps.add(squared, mSq, (float)1e-8);
			TensorOps.cdiv(squared, deltaSq, squared);
			
			TensorOps.sqrt(squared, squared);
			TensorOps.cmul(deltaParams, deltaParams, squared);
			TensorOps.mul(deltaParams, deltaParams, -1);
			
			
			// calculate new mean delta squared
			deltaSq = TensorOps.mul(deltaSq, deltaSq, decayRate);
			deltaParams.copyInto(squared);
			squared = TensorOps.cmul(squared, squared, squared);
			deltaSq = TensorOps.add(deltaSq, deltaSq, (1-decayRate), squared);
			
			// set DeltaParameters to be sure in case of remote module instance
			e.getValue().setDeltaParameters(deltaParams);
		});
	}

}
