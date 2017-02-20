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


public class Average extends Join {

	public Average() {
		super();
	}
	
	public Average(UUID id) {
		super(id);
	}
	
	public Average(boolean waitForAll) {
		super(waitForAll);
	}
	
	public Average(UUID id, boolean waitForAll) {
		super(id, waitForAll);
	}
	
	@Override
	protected void forward() {
		// element-wise average over inputs
		if(output==null){
			output = new Tensor(inputs.values().stream().filter(t -> t !=null).findFirst().get().dims());
		}
		output.fill(0.0f);
		for(Tensor t : inputs.values()){
			if(t!=null)
				output = TensorOps.add(output, output, t);
		}
		output = TensorOps.div(output, output, inputs.size());
	}

	@Override
	protected void backward() {
		// forward same error to all
		for(UUID id : gradInputs.keySet()){
			Tensor gradInput = gradInputs.get(id);
			gradInput = gradOutput.copyInto(gradInput);
			gradInput = TensorOps.div(gradInput, gradInput, gradInputs.size());
			gradInputs.put(id, gradInput);
		}
	}

}
