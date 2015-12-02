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
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Frame extends AbstractModule {

	// dims of the scaled tensor
	private final int[] targetDims;
	
	public Frame(TensorFactory factory, final int... dims){
		super(factory);
		this.targetDims = dims;
	}
	
	public Frame(TensorFactory factory, UUID id, final int... dims){
		super(factory, id);
		this.targetDims = dims;
	}

	@Override
	protected void forward() {
		
		int targetDim = targetDims.length;
		
		int inputDim = input.dim();
		int[] inputDims = input.dims();
		
		Tensor in = input;
		if(targetDim < inputDim){
			// from 3 to 2D -> select first, remove dimension 
			inputDim = 2;
			inputDims = new int[]{inputDims[1], inputDims[2]};
			in = input.select(0, 0);
		} else if(inputDim < targetDim){
			// from 2 to 3D -> reshape first, add dimension
			in.reshape(1, inputDims[0], inputDims[1]);
			inputDim = 3;
			inputDims = new int[]{1, inputDims[0], inputDims[1]};
		}
		
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
			if(inputDims[i]-narrowDims[i] < 0){
				// input smaller then narrowing ... just keep input
				// could be the case when going from 2D to 3D
				ranges[i*2] = 0;
				ranges[i*2+1] = inputDims[i];
			} else {
				ranges[i*2] = (inputDims[i]-narrowDims[i])/2;
				ranges[i*2+1] = narrowDims[i];
			}
		}
		
		Tensor narrowed = in.narrow(ranges);
		output = factory.getTensorMath().scale2D(output, narrowed, targetDims);
	}

	@Override
	protected void backward() {
		// not implemented
	}

}
