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

import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.nn.learn.processors.config.AdagradConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class AdagradProcessor extends GradientProcessor {

	private final AdagradConfig config;
	
	private final Map<UUID, Tensor> accumulatedSquared = new HashMap<>();
	private Tensor squared = null;
	
	public AdagradProcessor(NeuralNetwork nn, AdagradConfig config ) {
		super(nn);
		
		this.config = config;
	}
	
	@Override
	public void updateDelta(long i) {
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor deltaParams = e.getValue().getDeltaParameters();
			
			// accumulated squared gradients
			squared = deltaParams.copyInto(squared);
			squared = TensorOps.cmul(squared, squared, squared);
			
			Tensor accSq = accumulatedSquared.get(e.getKey());
			if(accSq == null){
				accSq = squared.copyInto(accSq);
			} else {
				accSq = TensorOps.add(accSq, accSq, squared);
			}
			accumulatedSquared.put(e.getKey(), accSq);

			// deltaparams = - learning_rate * dx / np.sqrt(accSq + epsilon)
			TensorOps.mul(deltaParams, deltaParams, -config.learningRate);
			
			// add 1e-8 to avoid div by zero, reuse squared tensor for this
			TensorOps.add(squared, accSq, config.epsilon);
			
			// now div by cached tensor
			TensorOps.cdiv(deltaParams, deltaParams, squared);
			
			// set DeltaParameters to be sure in case of remote module instance
			e.getValue().setDeltaParameters(deltaParams);
		});
	}

}
