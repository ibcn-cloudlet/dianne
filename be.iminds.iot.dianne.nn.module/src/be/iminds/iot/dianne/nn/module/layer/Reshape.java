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
package be.iminds.iot.dianne.nn.module.layer;

import java.util.Arrays;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;

public class Reshape extends AbstractModule {

	private final int[] targetDims;
	private final int targetSize;
	
	private int[] inputDims;
	
	public Reshape(final int... dims){
		super();
		this.targetDims = dims;
		int size = 1;
		for(int t : targetDims){
			size *= t;
		}
		targetSize = size;
	}
	
	public Reshape(UUID id, final int... dims){
		super(id);
		this.targetDims = dims;
		int size = 1;
		for(int t : targetDims){
			size *= t;
		}
		targetSize = size;
	}

	@Override
	protected void forward() {
		inputDims = input.dims();
		int inputSize = input.size();
		
		output = input.copyInto(output);
		
		if(inputSize != targetSize){
			if(inputSize / inputDims[0] == targetSize){
				// batch dimension
				int[] newDim = new int[targetDims.length + 1];
				newDim[0] = inputDims[0];
				for(int i=0;i<targetDims.length;i++){
					newDim[i+1] = targetDims[i];
				}
				output.reshape(newDim);
			} else {
				throw new RuntimeException("Invalid input dimensions to reshape?! input dims : " +Arrays.toString(input.dims())+" target dims: "+Arrays.toString(targetDims));
			}
		} else {
			output.reshape(targetDims);
		}
	}

	@Override
	protected void backward() {
		gradInput = gradOutput.copyInto(gradInput);
		gradInput.reshape(inputDims);
	}

}
