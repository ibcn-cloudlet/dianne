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

import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Concat extends Join {

	public Concat(TensorFactory factory) {
		super(factory);
	}
	
	public Concat(TensorFactory factory, UUID id) {
		super(factory, id);
	}
	
	@Override
	protected void forward() {
		// concat from N parts in dimension 0 (other dims should be equal)
		if(prev!=null){
			int[] dims = null;
			for(Tensor i : inputs.values()){
				if(i!=null){
					if(dims==null){
						dims = i.dims();
					} else {
						dims[0] += i.dims()[0];
					}
				} else {
					// what with null inputs? exception or skip?
				}
			}
	
			if(output==null || !output.hasDim(dims)){
				output = factory.createTensor(dims);
			}
		
			int offset = 0;
			for(int i=0;i<prev.length;i++){
				Tensor in = inputs.get(prevIds[i]);
				if(in!=null){
					int size = in.dims()[0];
					in.copyInto(output.narrow(0, offset, size));
					offset+=size;
				}
			}
		}
	}

	@Override
	protected void backward() {
		if(prev!=null){
			int offset = 0;
			for(int i=0;i<prev.length;i++){
				Tensor in = inputs.get(prevIds[i]);
				if(in!=null){
					int size = in.dims()[0];
					gradInputs.put(prevIds[i], gradOutput.narrow(0, offset, size));
					offset+=size;
				}
			}
		}
	}

}
