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

public class Zeropad extends AbstractModule {

	private final int[] padding;
	
	public Zeropad(final int... padding){
		super();
		for(int i = 0;i<padding.length;i++){
			if(padding[i]!=0){
				this.padding = padding;
				return;
			}
		}
		this.padding = null;
	}
	
	public Zeropad(UUID id, final int... padding){
		super(id);
		for(int i = 0;i<padding.length;i++){
			if(padding[i]!=0){
				this.padding = padding;
				return;
			}
		}
		this.padding = null;
	}

	private int[] inputDims;
	private int[] outputDims;
	private int[] ranges;
	
	@Override
	protected void forward() {
		// pass through if no padding specified
		if(padding == null){
			output = input;
			return;
		}
		
		inputDims = input.dims();
		outputDims = new int[inputDims.length];
		// apply padding from back to front ... 2d padding on 3d tensor means padding the last 2 dims
		ranges = new int[outputDims.length*2];
		for(int i=0;i<outputDims.length;i++){
			int pad = padding.length > i ? padding[padding.length-1-i] : 0;
			outputDims[outputDims.length-1-i] = inputDims[outputDims.length-1-i] + 2*pad;
			
			ranges[(outputDims.length-1-i)*2] = pad;
			ranges[(outputDims.length-1-i)*2 + 1] = inputDims[outputDims.length-1-i];
		}
		if(output==null){
			output = new Tensor(outputDims);
		}
		output.reshape(outputDims);
		output.fill(0.0f);
		
		Tensor narrowed = output.narrow(ranges); 
		input.copyInto(narrowed);
	}

	@Override
	protected void backward() {
		// pass through if no padding specified
		if(padding==null){
			gradInput = gradOutput;
			return;
		}
		
		gradOutput.reshape(outputDims);
		if(gradInput == null){
			gradInput = new Tensor(inputDims);
		}
		gradInput.reshape(inputDims);
		Tensor narrowed = gradOutput.narrow(ranges);
		narrowed.copyInto(gradInput);
	}

}
