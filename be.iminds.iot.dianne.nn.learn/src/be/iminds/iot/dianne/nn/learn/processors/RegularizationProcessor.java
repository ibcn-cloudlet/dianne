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

import be.iminds.iot.dianne.api.nn.learn.GradientProcessor;
import be.iminds.iot.dianne.tensor.Tensor;

public class RegularizationProcessor extends GradientProcessor {

	private final float rate;
	
	public RegularizationProcessor(GradientProcessor p, float rate) {
		super(p);
		this.rate = rate;
	}
	
	@Override
	public void updateDelta(long i) {
		nn.getTrainables().values().stream().forEach(m -> {
			// Get the gradients and parameters
			Tensor deltaParams = m.getDeltaParameters();
			Tensor params = m.getParameters();
			
			// Subtract previous parameters
			factory.getTensorMath().sub(deltaParams, deltaParams, rate, params);
			
			// Set DeltaParameters to be sure in case of remote module instance
			m.setDeltaParameters(deltaParams);
		});
	}
}
