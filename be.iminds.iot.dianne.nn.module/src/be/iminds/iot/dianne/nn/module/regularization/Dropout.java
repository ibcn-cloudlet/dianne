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
package be.iminds.iot.dianne.nn.module.regularization;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class Dropout extends AbstractModule {

	private float p;
	
	private Tensor mask = null;
	
	public Dropout(float p){
		super();
		this.p = 1-p; // p = rate to drop, so 1-p is bernoulli parameter
	}
	
	public Dropout(UUID id, float p){
		super(id);
		this.p = 1-p;
	}
	
	@Override
	protected void forward() {
		if(mask == null || !mask.hasDim(input.dims())){
			mask = new Tensor(input.dims());
		}
		
		if(p < 1 && train) {
		
			mask.bernoulli(p);
			TensorOps.div(mask, mask, p);
		
			output = TensorOps.cmul(output, input, mask);
		
		} else {
			// just forward
			output = input;
		}
	}

	@Override
	protected void backward() {
		if(p < 1 ){
			gradInput = TensorOps.cmul(gradInput, gradOutput, mask);
		} else {
			gradInput = gradOutput;
		}
	}

	@Override
	public void setProperty(String key, Object val){
		if(key.equals("rate")){
			p = 1-Float.parseFloat(val.toString());
		} else {
			super.setProperty(key, val);
		}
	}
}
