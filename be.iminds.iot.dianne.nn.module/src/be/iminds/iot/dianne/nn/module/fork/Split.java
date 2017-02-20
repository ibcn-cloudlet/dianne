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

public class Split extends Fork {

	// dimension on which to split
	// counted starting from the last dimension
	// e.g. 3-dim tensor (x,y,z), dim 0 = split on z, dim 1 = split on y
	// this allows to have the desired split in case of batched tensors
	private final int dim;
	
	public Split(int dim) {
		super();
		this.dim = dim;
	}
	
	public Split(UUID id, int dim) {
		super(id);
		this.dim = dim;
	}
	
	public Split(boolean waitForAll, int dim) {
		super(waitForAll);
		this.dim = dim;
	}
	
	public Split(UUID id, boolean waitForAll, int dim) {
		super(id, waitForAll);
		this.dim = dim;
	}
	
	@Override
	protected void forward() {
		// split in N equal parts in dimension 1
		int[] inputDims = input.dims();
		int splitDim = inputDims.length-1-dim;
		if(next!=null){
			int size = inputDims[splitDim]/next.length;
			for(int i=0;i<next.length;i++){
				outputs.put(nextIds[i], input.narrow(splitDim, i*size, size));
			}
		}
	}

	@Override
	protected void backward() {
		if(next!=null){
			int[] dims = gradOutputs.values().iterator().next().dims();
			int splitDim = dims.length-1-dim;
			int size = dims[splitDim];
			if(gradInput==null){
				dims[splitDim] = dims[splitDim]*gradOutputs.size();
				gradInput = new Tensor(dims);
			}

			for(int i=0;i<next.length;i++){
				gradOutputs.get(nextIds[i]).copyInto(gradInput.narrow(splitDim, i*size, size));
			}
		}
		
	}

}
