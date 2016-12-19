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

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.Tensor;

public class Narrow extends AbstractModule {

	// narrow ranges - should match the dimensions though
	private final int[] ranges;
	
	private int[] inputDims;
	private int batched = 0;
	
	public Narrow(final int... ranges){
		super();
		this.ranges = ranges;
	}
	
	public Narrow(UUID id, final int... ranges){
		super(id);
		this.ranges = ranges;
	}

	@Override
	protected void forward() {
		inputDims = input.dims();
		if(inputDims.length == ranges.length/2 + 1){
			// batched input
			batched = 1;
		}  else {
			batched = 0;
		}
	
		output = input;
		for(int i=0;i<ranges.length-1;i+=2){
			output = output.narrow(i/2 + batched, ranges[i], ranges[i+1]); 
		}
	
	}

	@Override
	protected void backward() {
		if(gradInput == null){
			gradInput = new Tensor(inputDims);
		}
		gradInput.reshape(inputDims);
		gradInput.fill(0.0f);	
		
		Tensor narrowedGradIn = gradInput;
		for(int i=0;i<ranges.length-1;i+=2){
			narrowedGradIn = narrowedGradIn.narrow(i/2 + batched, ranges[i], ranges[i+1]); 
		}
		
		gradOutput.copyInto(narrowedGradIn);
	}

}
