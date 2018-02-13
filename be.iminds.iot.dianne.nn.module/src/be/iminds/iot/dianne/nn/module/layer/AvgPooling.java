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
import be.iminds.iot.dianne.tensor.ModuleOps;

public class AvgPooling extends AbstractModule {
	
	private int width;
	private int height;
	private int depth;
	
	private int strideX = 1;
	private int strideY = 1;
	private int strideZ = 1;
	
	// unused for now
	private int padX = 0;
	private int padY = 0;
	private int padZ = 0;
	
	private boolean ceil = false;
	private boolean include_pad = false;
	
	private Type type;
	
	/* Temporal constructors */
	public AvgPooling(int width, int stride){
		super();
		this.width = width;
		this.strideX = stride;
		type = Type.TEMPORAL;
	}
	
	public AvgPooling(UUID id,
			 int width, int stride){
		super(id);
		this.width = width;
		this.strideX = stride;
		type = Type.TEMPORAL;
	}
	
	public AvgPooling(int width, int stride, int pad){
		super();
		this.width = width;
		this.strideX = stride;
		this.padX = pad;
		type = Type.TEMPORAL;
	}
	
	public AvgPooling(UUID id,
			 int width, int stride, int pad){
		super(id);
		this.width = width;
		this.strideX = stride;
		this.padX = pad;
		type = Type.TEMPORAL;
	}
	
	/* Spatial constructors */
	public AvgPooling(int width, int height, int strideX, int strideY){
		super();
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
		type = Type.SPATIAL;
	}
	
	public AvgPooling(UUID id,
			 int width, int height, int strideX, int strideY){
		super(id);
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
		type = Type.SPATIAL;
	}
	
	public AvgPooling(int width, int height, int strideX, int strideY, int padX, int padY){
		super();
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
		this.padX = padX;
		this.padY = padY;
		type = Type.SPATIAL;
	}
	
	public AvgPooling(UUID id,
			 int width, int height, int strideX, int strideY, int padX, int padY){
		super(id);
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
		this.padX = padX;
		this.padY = padY;
		type = Type.SPATIAL;
	}
	
	/* Volumetric constructors */
	public AvgPooling(int width, int height, int depth, 
			int strideX, int strideY, int strideZ,
			int padX, int padY, int padZ){
		super();
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.strideX = strideX;
		this.strideY = strideY;
		this.strideZ = strideZ;
		this.padX = padX;
		this.padY = padY;
		this.padZ = padZ;
		type = Type.VOLUMETRIC;
	}
	
	public AvgPooling(UUID id,
			 int width, int height, int depth, 
			 int strideX, int strideY, int strideZ,
			 int padX, int padY, int padZ){
		super(id);
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.strideX = strideX;
		this.strideY = strideY;
		this.strideZ = strideZ;
		this.padX = padX;
		this.padY = padY;
		this.padZ = padZ;
		type = Type.VOLUMETRIC;
	}

	@Override
	protected void forward() {
		switch(type){
		case TEMPORAL:
			// temporal avg pooling as spatial pooling with height = 1
			input.reshape(input.dims(), 1);
			output = ModuleOps.spatialavgpool(output, input, 1, 
					width == -1 ? input.dims()[input.dim()-1] : width, 1, strideX, 0, padX, ceil, include_pad);
			int[] outputDims = output.dims();
			output.reshape(Arrays.copyOf(outputDims, outputDims.length-1));
			break;
		case SPATIAL:
			output = ModuleOps.spatialavgpool(output, input, 
					width == -1 ? input.dims()[input.dim()-1] : width,
					height == -1 ? input.dims()[input.dim()-2] : height, strideX, strideY, padX, padY, ceil, include_pad);
			break;
		case VOLUMETRIC:
			output = ModuleOps.volumetricavgpool(output, input,
					width == -1 ? input.dims()[input.dim()-1] : width,
					height == -1 ? input.dims()[input.dim()-2] : height, 
					depth == -1 ? input.dims()[input.dim()-3] : depth, strideX, strideY, strideZ);
			break;
		}
	}

	@Override
	protected void backward() {	
		switch(type){
		case TEMPORAL:
			// 1D as 2d with height = 1
			gradOutput.reshape(output.dims(), 1);
			gradInput = ModuleOps.spatialavgpoolGradIn(gradInput, gradOutput, input, output, 1, width, 1, strideX, 0, padX, ceil, include_pad);
			int[] inputDims = input.dims();
			gradInput.reshape(Arrays.copyOf(inputDims, inputDims.length-1));
			break;
		case SPATIAL:
			gradInput = ModuleOps.spatialavgpoolGradIn(gradInput, gradOutput, input, output, width, height, strideX, strideY, padX, padY, ceil, include_pad);
			break;
		case VOLUMETRIC:
			gradInput = ModuleOps.volumetricavgpoolGradIn(gradInput, gradOutput, input, output, width, height, depth, strideX, strideY, strideZ);
			break;
		}
	}
	
}
