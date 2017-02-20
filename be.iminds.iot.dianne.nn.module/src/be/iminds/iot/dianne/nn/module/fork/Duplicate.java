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
package be.iminds.iot.dianne.nn.module.fork;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.Fork;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;


public class Duplicate extends Fork {

	public Duplicate() {
		super();
	}
	
	public Duplicate(UUID id) {
		super(id);
	}
	
	public Duplicate(boolean waitForAll) {
		super(waitForAll);
	}
	
	public Duplicate(UUID id, boolean waitForAll) {
		super(id, waitForAll);
	}

	@Override
	protected void forward() {
		// just duplicate output
		for(UUID id : outputs.keySet()){
			outputs.put(id, input.copyInto(outputs.get(id)));
		}
	}

	@Override
	protected void backward() {
		// accumulate gradOutputs in gradInput
		if(gradInput==null){
			gradInput = new Tensor(input.dims());
		}
		gradInput.fill(0.0f);
		for(Tensor t : gradOutputs.values()){
			if(t!=null)
				gradInput = TensorOps.add(gradInput, gradInput, t);
		}
	}
	
}
