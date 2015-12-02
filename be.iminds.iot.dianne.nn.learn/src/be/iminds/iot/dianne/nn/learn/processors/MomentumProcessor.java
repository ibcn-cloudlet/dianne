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

import be.iminds.iot.dianne.api.nn.learn.Processor;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Additional learning techniques like Momentum can be implemented as a Processor decorator
 */
public class MomentumProcessor extends AbstractProcessor {

	private final Processor decorated;
	private final float momentum;
	
	private Map<UUID, Tensor> previousDelta = new HashMap<UUID, Tensor>();
	
	public MomentumProcessor( AbstractProcessor p, float momentum ) {
		super(p.factory, p.nn, p.logger);
		this.decorated = p;
		
		this.momentum = momentum;
	}
	
	@Override
	public float processNext() {
		float error = decorated.processNext();
		
		// add momentum
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor prev = previousDelta.get(e.getKey());
			if(prev!=null){
				Tensor deltaParams = e.getValue().getDeltaParameters();
				factory.getTensorMath().add(deltaParams, deltaParams , momentum, prev);
			}
		});
		
		// copy to previousGrad
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor prev = previousDelta.get(e.getKey());
			Tensor deltaParams = e.getValue().getDeltaParameters();
			deltaParams.copyInto(prev);
			previousDelta.put(e.getKey(), deltaParams);
		});
		
		return error;
	}

}
