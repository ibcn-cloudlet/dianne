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

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Additional learning techniques like Momentum can be implemented as a Processor decorator
 */
public class RegularizationProcessor extends AbstractProcessor {

	private float regularization = 0.001f;
	
	public RegularizationProcessor( AbstractProcessor p, float regularization) {
		super(p);
		this.regularization = regularization;
	}
	
	@Override
	public float processNext(float error) {
		// subtract previous parameters
		nn.getTrainables().entrySet().stream().forEach(e -> {
			Tensor params = e.getValue().getParameters();
			Tensor deltaParams = e.getValue().getDeltaParameters();
			factory.getTensorMath().sub(deltaParams, deltaParams, regularization, params);
			
			// set DeltaParameters to be sure in case of remote module instance
			e.getValue().setDeltaParameters(deltaParams);
		});
		
		return error;
	}

}
