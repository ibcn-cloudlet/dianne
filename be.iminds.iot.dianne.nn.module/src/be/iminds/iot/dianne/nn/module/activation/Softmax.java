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
package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Softmax extends AbstractModule {

	private float alpha = 0.0001f;
	
	public Softmax(TensorFactory factory) {
		super(factory);
	}
	
	public Softmax(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().softmax(output, input);
		
		// this makes sure that you don't end up with zeros and a one, which 
		// gives -Inf in the NLL ... this does add a (small) error though...
		output = factory.getTensorMath().add(output, output, alpha);
		output = factory.getTensorMath().div(output, output, 1f + alpha*output.size());
	}

	@Override
	protected void backward() {
		float sum = factory.getTensorMath().sum(factory.getTensorMath().cmul(null, gradOutput, output));
		
		gradInput = factory.getTensorMath().sub(gradInput, gradOutput, sum);
		gradInput = factory.getTensorMath().cmul(gradInput, output, gradInput);
	}
}
