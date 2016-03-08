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
package be.iminds.iot.dianne.nn.module.preprocessing;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Narrow extends AbstractModule {

	// narrow ranges - should match the dimensions though
	private final int[] ranges;
	
	public Narrow(TensorFactory factory, final int... ranges){
		super(factory);
		this.ranges = ranges;
	}
	
	public Narrow(TensorFactory factory, UUID id, final int... ranges){
		super(factory, id);
		this.ranges = ranges;
	}

	@Override
	protected void forward() {
		output = input.narrow(ranges);
	}

	@Override
	protected void backward() {
		// not implemented
	}

}
