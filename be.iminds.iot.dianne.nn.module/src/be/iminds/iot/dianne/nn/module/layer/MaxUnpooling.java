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

import be.iminds.iot.dianne.nn.module.join.Join;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;

public class MaxUnpooling extends Join {
	
	private int width;
	private int height;
	private int depth;
	
	private int strideX;
	private int strideY;
	private int strideZ;
	
	// unused for now
	private int padX = 0;
	private int padY = 0;
	private int padZ = 0;
	
	private Type type;
	
	// indices should be given by the corresponding MaxPooling module
	private Tensor indices;
	
	/* Temporal constructors */
	public MaxUnpooling(int width, int stride){
		super();
		this.width = width;
		this.strideX = stride;
		type = Type.TEMPORAL;
	}
	
	public MaxUnpooling(UUID id,
			 int width, int stride){
		super(id);
		this.width = width;
		this.strideX = stride;
		type = Type.TEMPORAL;
	}
	
	/* Spatial constructors */
	public MaxUnpooling(int width, int height, int strideX, int strideY){
		super();
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
		type = Type.SPATIAL;
	}
	
	public MaxUnpooling(UUID id,
			 int width, int height, int strideX, int strideY){
		super(id);
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
		type = Type.SPATIAL;
	}
	
	/* Volumetric constructors */
	public MaxUnpooling(int width, int height, int depth, int strideX, int strideY, int strideZ){
		super();
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.strideX = strideX;
		this.strideY = strideY;
		this.strideZ = strideZ;
		type = Type.VOLUMETRIC;
	}
	
	public MaxUnpooling(UUID id,
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


	@Override
	protected void forward() {
		if(prev==null || prevIds.length!=2){
			throw new RuntimeException("MaxUnPooling not configured correctly, should receive both input and indices");
		}
		
		input = inputs.get(prevIds[0]);
		indices = inputs.get(prevIds[1]);
		if(indices.size() == 0){
			throw new RuntimeException("Invalid indices tensor provided");
		}
		
		switch(type){
		case TEMPORAL:
			// TODO can we implement this with spatial variant?
			throw new UnsupportedOperationException();
		case SPATIAL:
			output = ModuleOps.spatialmaxunpool(output, input, indices, width, height, strideX, strideY, 0, 0);
			break;
		case VOLUMETRIC:
			output = ModuleOps.volumetricmaxunpool(output, input, indices, 
					width, height, depth, strideX, strideY, strideZ, padX, padY, padZ);
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
			// TODO can we implement this with spatial variant?
			throw new UnsupportedOperationException();
		case SPATIAL:
			gradInput = ModuleOps.spatialmaxunpoolGradIn(gradInput, gradOutput, input, indices, width, height, strideX, strideY, 0, 0);
			break;
		case VOLUMETRIC:
			gradInput = ModuleOps.volumetricmaxunpoolGradIn(gradInput, gradOutput, input, indices,
					width, height, depth, strideX, strideY, strideZ, padX, padY, padZ);
			break;
		}
		
		gradInputs.put(prevIds[0], gradInput);
	}
	
}
