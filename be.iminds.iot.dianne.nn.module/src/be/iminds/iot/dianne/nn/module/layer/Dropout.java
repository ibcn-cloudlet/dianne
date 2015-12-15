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
package be.iminds.iot.dianne.nn.module.layer;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Dropout extends AbstractModule {

	private final float p;
	
	private Tensor mask = null;
	
	public Dropout(TensorFactory factory, float p){
		super(factory);
		this.p = 1-p; // p = rate to drop, so 1-p is bernoulli parameter
	}
	
	public Dropout(TensorFactory factory, UUID id, float p){
		super(factory, id);
		this.p = 1-p;
	}
	
	@Override
	protected void forward() {
		if(mask == null || !mask.hasDim(input.dims())){
			mask = factory.createTensor(input.dims());
		}
		
		mask.bernoulli(p);
		factory.getTensorMath().div(mask, mask, p);
		
		output = factory.getTensorMath().cmul(output, input, mask);
	}

	@Override
	protected void backward() {
		gradInput = factory.getTensorMath().cmul(gradInput, gradOutput, mask);
	}

}
