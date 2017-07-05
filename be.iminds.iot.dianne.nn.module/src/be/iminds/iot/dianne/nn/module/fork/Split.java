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
	
	private final int[] splits;
	
	public Split(int dim) {
		super();
		this.dim = dim;
		this.splits = null;
	}
	
	public Split(int dim, int[] splits) {
		super();
		this.dim = dim;
		this.splits = splits;
	}
	
	public Split(UUID id, int dim) {
		super(id);
		this.dim = dim;
		this.splits = null;
	}
	
	public Split(UUID id, int dim, int[] splits) {
		super(id);
		this.dim = dim;
		this.splits = splits;
	}
	
	
	public Split(boolean waitForAll, int dim) {
		super(waitForAll);
		this.dim = dim;
		this.splits = null;
	}

	public Split(boolean waitForAll, int dim, int[] splits) {
		super(waitForAll);
		this.dim = dim;
		this.splits = splits;
	}
	
	public Split(UUID id, boolean waitForAll, int dim) {
		super(id, waitForAll);
		this.dim = dim;
		this.splits = null;
	}
	
	public Split(UUID id, boolean waitForAll, int dim, int[] splits) {
		super(id, waitForAll);
		this.dim = dim;
		this.splits = splits;
	}
	
	@Override
	protected void forward() {
		// split in N equal parts in dimension 1
		int[] inputDims = input.dims();
		int splitDim = inputDims.length-1-dim;
		if(next!=null){
			if(splits == null){
				// all equal size
				int size = inputDims[splitDim]/next.length;
				for(int i=0;i<next.length;i++){
					outputs.put(nextIds[i], input.narrow(splitDim, i*size, size));
				}
			} else {
				int start = 0;
				int size = splits[0];
				for(int i=0;i<next.length;i++){
					outputs.put(nextIds[i], input.narrow(splitDim, start, size));
					start += size;
					if(i+1 < splits.length){
						size = splits[i+1] - start;
					} else {
						size = inputDims[splitDim] - start;
					}
				}
			}
		}
	}

	@Override
	protected void backward() {
		if(next!=null){
			int[] dims = gradOutputs.values().iterator().next().dims();
			int splitDim = dims.length-1-dim;
			if(gradInput==null){
				dims[splitDim] = 0;
				for(Tensor gradOut : gradOutputs.values()){
					dims[splitDim] += gradOut.dims()[splitDim];
				}
				gradInput = new Tensor(dims);
			}
			int[] inputDims = gradInput.dims();

			if(splits == null){
				// all equal size
				for(int i=0;i<next.length;i++){
					Tensor gradOut = gradOutputs.get(nextIds[i]);
					int size = gradOut.dims()[splitDim];
					gradOut.copyInto(gradInput.narrow(splitDim, i*size, size));
				}
			} else {
				int start = 0;
				int size = splits[0];
				for(int i=0;i<next.length;i++){
					gradOutputs.get(nextIds[i]).copyInto(gradInput.narrow(splitDim, start, size));
					start += size;
					if(i+1 < splits.length){
						size = splits[i+1] - start;
					} else {
						size = inputDims[splitDim] - start;
					}
				}
			}
		}
		
	}

}
