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
import be.iminds.iot.dianne.nn.module.ModuleOps;

public class Tanh extends AbstractModule {

	public Tanh() {
		super();
	}
	
	public Tanh(UUID id) {
		super(id);
	}
	
	@Override
	protected void forward() {
		output = ModuleOps.tanh(output, input);
	}

	@Override
	protected void backward() {
		// derivative of tanh:
		// dtanh/dx = 1-tanh^2 
		//
		// thus:
		// gradInput = gradOutput * ( dtan/dx(input) )
		//           = gradOutput * (1 - tanh^2(input))
		//           = gradOutput * (1 - output^2)
		
		// TODO do this in one operation
		//gradInput = TensorOps.cmul(gradInput, gradOutput, 
		//		TensorOps.dtanh(gradInput, output));
		gradInput = ModuleOps.tanhDin(gradInput, gradOutput, output);
	}

}
