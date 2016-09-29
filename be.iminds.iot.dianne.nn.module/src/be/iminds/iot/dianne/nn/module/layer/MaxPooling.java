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

import be.iminds.iot.dianne.nn.module.fork.Fork;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;

public class MaxPooling extends Fork {
	
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
	
	// temp tensor with max indices to speed up backward
	private Tensor indices = new Tensor();
	
	/* Temporal constructors */
	public MaxPooling(int width, int stride){
		super();
		this.width = width;
		this.strideX = stride;
		type = Type.TEMPORAL;
	}
	
	public MaxPooling(UUID id,
			 int width, int stride){
		super(id);
		this.width = width;
		this.strideX = stride;
		type = Type.TEMPORAL;
	}
	
	/* Spatial constructors */
	public MaxPooling(int width, int height, int strideX, int strideY){
		super();
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
		type = Type.SPATIAL;
	}
	
	public MaxPooling(UUID id,
			 int width, int height, int strideX, int strideY){
		super(id);
		this.width = width;
		this.height = height;
		this.strideX = strideX;
		this.strideY = strideY;
		type = Type.SPATIAL;
	}
	
	/* Volumetric constructors */
	public MaxPooling(int width, int height, int depth, int strideX, int strideY, int strideZ){
		super();
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.strideX = strideX;
		this.strideY = strideY;
		this.strideZ = strideZ;
		type = Type.VOLUMETRIC;
	}
	
	public MaxPooling(UUID id,
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
		switch(type){
		case TEMPORAL:
			output = ModuleOps.temporalmaxpool(output, input, indices, width, strideX);
			break;
		case SPATIAL:
			output = ModuleOps.spatialmaxpool(output, input, indices, width, height, strideX, strideY, 0, 0);
			break;
		case VOLUMETRIC:
			output = ModuleOps.volumetricmaxpool(output, input, indices, 
					width, height, depth, strideX, strideY, strideZ, padX, padY, padZ);
			break;
		}
		
		if(next!=null){
			outputs.put(nextIds[0], output);
			if(next.length==2){
				outputs.put(nextIds[1], indices);
			}
		}
	}

	@Override
	protected void backward() {	
		if(next==null){
			// in case of junit test mode there can be no next defined
			// then just pick whatever you find in gradOutputs
			gradOutput = gradOutputs.values().iterator().next();
		} else {
			gradOutput = gradOutputs.get(nextIds[0]);
		}
		
		if(gradInput == null){
			gradInput = new Tensor(input.dims());
		} 

		switch(type){
		case TEMPORAL:
			gradInput = ModuleOps.temporalmaxpoolGradIn(gradInput, gradOutput, input, output, indices, width, strideX);
			break;
		case SPATIAL:
			gradInput = ModuleOps.spatialmaxpoolGradIn(gradInput, gradOutput, input, output, indices, width, height, strideX, strideY, 0, 0);
			break;
		case VOLUMETRIC:
			gradInput = ModuleOps.volumetricmaxpoolGradIn(gradInput, gradOutput, input, output, indices,
					width, height, depth, strideX, strideY, strideZ, padX, padY, padZ);
			break;
		}
	}

}
