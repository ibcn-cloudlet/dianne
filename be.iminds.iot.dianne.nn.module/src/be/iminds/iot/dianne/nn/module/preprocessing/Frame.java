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
package be.iminds.iot.dianne.nn.module.preprocessing;

import java.util.Arrays;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class Frame extends AbstractModule {

	// dims of the scaled tensor
	private final int[] targetDims;
	
	private Tensor narrowed;
	
	public Frame(final int... dims){
		super();
		this.targetDims = dims;
		output = new Tensor();
	}
	
	public Frame(UUID id, final int... dims){
		super(id);
		this.targetDims = dims;
		output = new Tensor();
	}

	@Override
	protected void forward() {
		int targetDim = targetDims.length;
		int[] inputDims = input.dims();
		int inputDim = inputDims.length;
		
		int batches = 1;
		if(inputDim == targetDim + 1){
			batches = inputDims[0];
			int[] newDims = new int[inputDim-1];
			for(int i=0;i<newDims.length;i++){
				newDims[i] = inputDims[i+1];
			}
			inputDims = newDims;
			inputDim = newDims.length;
		} else if(inputDim == targetDim) {
			int[] newDims = new int[inputDim + 1];
			newDims[0] = 1;
			for(int i=0;i<inputDims.length;i++){
				newDims[i+1] = inputDims[i];
			}
			input.reshape(newDims); // add batch dimension of 1
		}
		
		int[] outputDims = new int[targetDim + 1];
		outputDims[0] = batches;
		for(int i=0;i<targetDims.length;i++){
			outputDims[i+1] = targetDims[i];
		}
		output.reshape(outputDims);
		
		for(int b=0;b<batches;b++){
			Tensor in = input.select(0, b);
			
			float sx = (float)inputDims[inputDim-1]/targetDims[targetDim-1];
			float sy = (float)inputDims[inputDim-2]/targetDims[targetDim-2];
			
			float s = sx < sy ? sx : sy;
			
			int[] narrowDims = new int[targetDim];
			for(int i=0;i<targetDim;i++){
				narrowDims[i] = targetDims[i];
			}
			narrowDims[targetDim-1] = (int) (targetDims[targetDim-1]*s);
			narrowDims[targetDim-2] = (int) (targetDims[targetDim-2]*s);
			
			int[] ranges = new int[targetDim*2];
			for(int i=0;i<targetDim;i++){
				ranges[i*2] = (inputDims[i]-narrowDims[i])/2;
				ranges[i*2+1] = narrowDims[i];
			}
			
			narrowed = in.narrow(ranges);
			TensorOps.scale2D(output.select(0, b), narrowed, targetDims);
		}
		
		if(batches==1){
			output.reshape(targetDims);
		}
	}

	@Override
	protected void backward() {
		// not implemented
	}

}
