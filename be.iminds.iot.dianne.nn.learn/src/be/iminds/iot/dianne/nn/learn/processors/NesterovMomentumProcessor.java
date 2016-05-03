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

import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Additional learning techniques like Momentum can be implemented as a Processor decorator
 */
public class NesterovMomentumProcessor extends GradientProcessor {

	private final float momentum;
	
	private Map<UUID, Tensor> previousVelocity = new HashMap<UUID, Tensor>();
	private Map<UUID, Tensor> velocity = new HashMap<UUID, Tensor>();
	
	public NesterovMomentumProcessor( GradientProcessor p, float momentum ) {
		super(p);
		this.momentum = momentum;
	}
	
	@Override
	public void updateDelta(long i) {
		// copy to previous velocity
		velocity.entrySet().stream().forEach(e -> {
			Tensor v = e.getValue();
			if(v!=null){
				Tensor prev = previousVelocity.get(e.getKey());
				previousVelocity.put(e.getKey(), v.copyInto(prev));
			}
		});
		
		// update velocity
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor deltaParams = e.getValue().getDeltaParameters();
			Tensor v = velocity.get(e.getKey());
			if(v!=null){
				v = TensorOps.add(v, deltaParams, momentum , v);
			} else {
				// if no velocity yet, just copy deltaParams
				v = deltaParams.copyInto(v);
			}
			velocity.put(e.getKey(), v);
		});
		
		// update delta parameters
		nn.getTrainables().entrySet().stream().forEach(e -> {
	
			// deltaParams = -momentum*v_prev + (1+momentum) * v (http://cs231n.github.io/neural-networks-3/)
			Tensor deltaParams = e.getValue().getDeltaParameters();
	
			Tensor prev = previousVelocity.get(e.getKey());
			if(prev!=null){
				TensorOps.mul(deltaParams, prev, -momentum);
			} else {
				deltaParams.fill(0.0f);
			}
			
			Tensor v = velocity.get(e.getKey());
			TensorOps.add(deltaParams, deltaParams , (1+momentum), v);
			
			// set DeltaParameters to be sure in case of remote module instance
			e.getValue().setDeltaParameters(deltaParams);
		});
	}

}
