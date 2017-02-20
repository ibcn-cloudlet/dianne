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

public class Concat extends Join {

	// dimension on which to concat
	// counted starting from the last dimension
	// e.g. 3-dim tensor (x,y,z), dim 0 = concat on z, dim 1 = concat on y
	// this allows to have the desired concatenation in case of batched tensors
	private final int dim;
	
	public Concat(int dim) {
		super();
		this.dim = dim;
	}
	
	public Concat(UUID id, int dim) {
		super(id);
		this.dim = dim;
	}
	
	public Concat(boolean waitForAll, int dim) {
		super(waitForAll);
		this.dim = dim;
	}
	
	public Concat(UUID id, boolean waitForAll, int dim) {
		super(id, waitForAll);
		this.dim = dim;
	}
	
	@Override
	protected void forward() {
		if(prev!=null){
			int[] dims = null;
			int concatDim = 0;
			for(Tensor i : inputs.values()){
				if(i!=null){
					if(dims==null){
						dims = i.dims();
						concatDim = dims.length-dim-1;
					} else {
						dims[concatDim] += i.dims()[concatDim];
					}
				} else {
					// what with null inputs? exception or skip?
				}
			}
	
			if(output==null){
				output = new Tensor(dims);
			}
			
			output.reshape(dims);
		
			int offset = 0;
			for(int i=0;i<prev.length;i++){
				Tensor in = inputs.get(prevIds[i]);
				if(in!=null){
					int size = in.dims()[concatDim];
					in.copyInto(output.narrow(concatDim, offset, size));
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
					int[] inputDims = in.dims();
					int concatDim = inputDims.length-1-dim;
					int size = inputDims[concatDim];
					gradInputs.put(prevIds[i], gradOutput.narrow(concatDim, offset, size));
					offset+=size;
				}
			}
		}
	}

}
