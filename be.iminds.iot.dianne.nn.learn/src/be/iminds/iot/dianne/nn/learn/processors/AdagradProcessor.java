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

public class AdagradProcessor extends AbstractProcessor {

	private final float learningRate;
	
	private final Map<UUID, Tensor> accumulatedSquared = new HashMap<>();
	
	public AdagradProcessor( AbstractProcessor p, float learningRate ) {
		super(p);
		
		this.learningRate = learningRate;
	}
	
	@Override
	public float processNext(float error) {

		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor deltaParams = e.getValue().getDeltaParameters();
			
			// accumulated squared gradients
			Tensor squared = deltaParams.copyInto(null);
			squared = factory.getTensorMath().cmul(squared, squared, squared);
			
			Tensor accSq = accumulatedSquared.get(e.getKey());
			if(accSq == null){
				accSq = squared.copyInto(accSq);
			} else {
				accSq = factory.getTensorMath().add(accSq, accSq, squared);
			}
			accumulatedSquared.put(e.getKey(), accSq);

			// deltaparams = - learning_rate * dx / np.sqrt(accSq + 1e-8)
			factory.getTensorMath().mul(deltaParams, deltaParams, -learningRate);
			
			// add 1e-8 to avoid div by zero, reuse squared tensor for this
			factory.getTensorMath().add(squared, accSq, (float) 1e-8);
			
			// now div by cached tensor
			factory.getTensorMath().cdiv(deltaParams, deltaParams, squared);
			
			// set DeltaParameters to be sure in case of remote module instance
			e.getValue().setDeltaParameters(deltaParams);
		});
		
		return error;
	}

}
