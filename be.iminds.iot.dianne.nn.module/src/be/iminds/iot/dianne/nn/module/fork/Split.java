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

import be.iminds.iot.dianne.tensor.Tensor;

public class Split extends Fork {

	public Split() {
		super();
	}
	
	public Split(UUID id) {
		super(id);
	}
	
	@Override
	protected void forward() {
		// split in N equal parts in dimension 1
		// TODO other split strategies?
		if(next!=null){
			int size = input.size(0)/next.length;
			for(int i=0;i<next.length;i++){
				outputs.put(nextIds[i], input.narrow(0, i*size, size));
			}
		}
	}

	@Override
	protected void backward() {
		if(next!=null){
			int[] dims = gradOutputs.values().iterator().next().dims();
			int size = dims[0];
			if(output==null){
				dims[0] = dims[0]*gradOutputs.size();
				gradInput = new Tensor(dims);
			}

			for(int i=0;i<next.length;i++){
				gradOutputs.get(nextIds[i]).copyInto(gradInput.narrow(0, i*size, size));
			}
		}
		
	}

}
