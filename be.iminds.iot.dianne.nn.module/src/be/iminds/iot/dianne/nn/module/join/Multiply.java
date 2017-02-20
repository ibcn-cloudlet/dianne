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
package be.iminds.iot.dianne.nn.module.join;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.Join;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;


public class Multiply extends Join {

	public Multiply() {
		super();
	}
	
	public Multiply(UUID id) {
		super(id);
	}
	
	public Multiply(boolean waitForAll) {
		super(waitForAll);
	}
	
	public Multiply(UUID id, boolean waitForAll) {
		super(id, waitForAll);
	}
	
	@Override
	protected void forward() {
		// elementwise multiply inputs
		if(output==null){
			output = new Tensor(inputs.values().stream().filter(t -> t !=null).findFirst().get().dims());
		}
		output.fill(1.0f);
		for(Tensor t : inputs.values()){
			if(t!=null)
				output = TensorOps.cmul(output, output, t);
		}
	}

	@Override
	protected void backward() {
		for(final UUID id : gradInputs.keySet()){
			Tensor gradInput = gradInputs.get(id);
			if(gradInput==null){
				gradInput = new Tensor(gradOutput.dims());
				gradInputs.put(id, gradInput);
			}
			gradOutput.copyInto(gradInput);
			
			final Tensor g = gradInput;
			inputs.entrySet().stream().filter(e -> !e.getKey().equals(id)).map(e -> e.getValue()).forEach(t -> TensorOps.cmul(g, g, t));
		}
	}

}
