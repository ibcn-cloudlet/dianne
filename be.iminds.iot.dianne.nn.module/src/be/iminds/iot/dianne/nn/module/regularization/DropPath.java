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

import java.util.Random;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.Tensor;

public class DropPath extends AbstractModule {

	private float p;
	private final Random r = new Random(System.currentTimeMillis());
	
	private boolean drop = false;
	
	public DropPath(float p){
		super();
		this.p = p;
	}
	
	public DropPath(UUID id, float p){
		super(id);
		this.p = p;
	}
	
	@Override
	protected void forward() {
		
		float f = r.nextFloat();
		if(f <= p){
			drop = true;
		} else {
			drop = false;
		}
		
		if(drop){
			if(output == null){
				output = new Tensor(input.dims());
			} else {
				output.reshape(input.dims());
			}
			output.fill(0.0f);
		} else {
			output = input.copyInto(output);
		}
	}

	@Override
	protected void backward() {
		if(drop){
			if(gradInput == null){
				gradInput = new Tensor(gradOutput.dims());
			} else {
				gradInput.reshape(gradOutput.dims());
			}
			gradInput.fill(0.0f);
		} else {
			gradInput = gradOutput.copyInto(gradInput); 
		}
	}

	@Override
	public void setProperty(String key, Object val){
		if(key.equals("rate")){
			p = Float.parseFloat(val.toString());
		} else {
			super.setProperty(key, val);
		}
	}
}
