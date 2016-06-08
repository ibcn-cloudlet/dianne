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

public class MaxPooling extends AbstractModule {
	
	private int w;
	private int h;
	private int d;
	
	private int sx;
	private int sy;
	private int sz;
	
	private Type type;
	
	// temp tensor with max indices to speed up backward
	private Tensor indices = new Tensor();
	
	/* Spatial constructors */
	public MaxPooling(int width, int height, int sx, int sy){
		super();
		this.w = width;
		this.h = height;
		this.sx = sx;
		this.sy = sy;
	}
	
	public MaxPooling(UUID id,
			 int width, int height, int sx, int sy){
		super(id);
		this.w = width;
		this.h = height;
		this.sx = sx;
		this.sy = sy;
	}

	@Override
	protected void forward() {
		output = ModuleOps.spatialmaxpool(output, input, indices, w, h, sx, sy, 0, 0);
	}

	@Override
	protected void backward() {	
		if(gradInput == null){
			gradInput = new Tensor(input.dims());
		} 

		ModuleOps.spatialmaxpoolGradIn(gradInput, gradOutput, input, indices, w, h, sx, sy, 0, 0);
	}
	
}
