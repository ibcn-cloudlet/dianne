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
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;

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
	
	/* Volumetric constructors */
	public AvgPooling(int width, int height, int depth, int strideX, int strideY, int strideZ){
		super();
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.strideX = strideX;
		this.strideY = strideY;
		this.strideZ = strideZ;
		type = Type.VOLUMETRIC;
	}
	
	public AvgPooling(UUID id,
			 int width, int height, int depth, int strideX, int strideY, int strideZ){
		super(id);
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.strideX = strideX;
		this.strideY = strideY;
		this.strideZ = strideZ;
		type = Type.VOLUMETRIC;
	}

	private int[] inputDims;
	
	@Override
	protected void forward() {
		switch(type){
		case TEMPORAL:
			// temporal avg pooling as spatial pooling in height dim?
			inputDims = input.dims();
			if(input.dim()==2){
				input.reshape(1, inputDims[0], inputDims[1]);
			}
			output = ModuleOps.spatialavgpool(output, input, 1, width, 1, strideX, 0, padX, ceil, include_pad);
			if(inputDims.length == 2){
				int[] outputDims = output.dims();
				output.reshape(outputDims[1], outputDims[2]);
			}
			break;
		case SPATIAL:
			output = ModuleOps.spatialavgpool(output, input, width, height, strideX, strideY, padX, padY, ceil, include_pad);
			break;
		case VOLUMETRIC:
			output = ModuleOps.volumetricavgpool(output, input, width, height, depth, strideX, strideY, strideZ);
			break;
		}
	}

	@Override
	protected void backward() {	
		if(gradInput == null){
			gradInput = new Tensor(input.dims());
		} 

		switch(type){
		case TEMPORAL:
			// 1D as 2d with height = 1
			gradInput = ModuleOps.spatialavgpoolGradIn(gradInput, gradOutput, input, width, 1, strideX, 1, padX, 0, ceil, include_pad);
			gradInput.reshape(inputDims);
			break;
		case SPATIAL:
			gradInput = ModuleOps.spatialavgpoolGradIn(gradInput, gradOutput, input, width, height, strideX, strideY, padX, padY, ceil, include_pad);
			break;
		case VOLUMETRIC:
			gradInput = ModuleOps.volumetricavgpoolGradIn(gradInput, gradOutput, input, width, height, depth, strideX, strideY, strideZ);
			break;
		}
	}
	
}
