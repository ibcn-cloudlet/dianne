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

import be.iminds.iot.dianne.tensor.Tensor;

public class AdadeltaProcessor extends AbstractProcessor {

	private final float decayRate;
	
	private final Map<UUID, Tensor> meanSquaredGradient = new HashMap<>();
	private final Map<UUID, Tensor> meanSquaredDelta = new HashMap<>();

	
	public AdadeltaProcessor( AbstractProcessor p, float decayRate) {
		super(p);
		
		this.decayRate = decayRate;
	}
	
	@Override
	public float processNext(float error) {

		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor deltaParams = e.getValue().getDeltaParameters();
			
			// calculate mean squared gradient
			Tensor s = deltaParams.copyInto(null);
			s = factory.getTensorMath().cmul(s, s, s);
			
			Tensor mSq = meanSquaredGradient.get(e.getKey());
			if(mSq == null){
				mSq = factory.getTensorMath().mul(mSq, s, (1-decayRate));
			} else {
				mSq = factory.getTensorMath().mul(mSq, mSq, decayRate);
				mSq = factory.getTensorMath().add(mSq, mSq, (1-decayRate), s);
			}
			meanSquaredGradient.put(e.getKey(), mSq);
			
			// calculate delta update
			Tensor deltaSq = meanSquaredDelta.get(e.getKey());
			if(deltaSq==null){
				deltaSq = factory.createTensor(deltaParams.dims());
				deltaSq.fill((float)1e-8);
				meanSquaredDelta.put(e.getKey(), deltaSq);
			} 
			
			// divide mean squared delta by mean squared gradient + 1e-8
			// update = - RMS(delta)/RMS(grad) * grad
			factory.getTensorMath().add(s, mSq, (float)1e-8);
			factory.getTensorMath().cdiv(s, deltaSq, s);
			
			factory.getTensorMath().sqrt(s, s);
			factory.getTensorMath().cmul(deltaParams, deltaParams, s);
			factory.getTensorMath().mul(deltaParams, deltaParams, -1);
			
			
			// calculate new mean delta squared
			deltaSq = factory.getTensorMath().mul(deltaSq, deltaSq, decayRate);
			deltaParams.copyInto(s);
			s = factory.getTensorMath().cmul(s, s, s);
			deltaSq = factory.getTensorMath().add(deltaSq, deltaSq, (1-decayRate), s);
			
			// set DeltaParameters to be sure in case of remote module instance
			e.getValue().setDeltaParameters(deltaParams);
		});
		
		return error;
	}

}
