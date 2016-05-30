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

public class SpatialAvgPooling extends AbstractModule {
	
	private int w;
	private int h;
	private int sx;
	private int sy;
	
	private boolean ceil = false;
	private boolean include_pad = false;
	
	public SpatialAvgPooling(int width, int height, int sx, int sy){
		super();
		this.w = width;
		this.h = height;
		this.sx = sx;
		this.sy = sy;
	}
	
	public SpatialAvgPooling(UUID id,
			 int width, int height, int sx, int sy){
		super(id);
		this.w = width;
		this.h = height;
		this.sx = sx;
		this.sy = sy;
	}

	@Override
	protected void forward() {
		int noPlanes = input.size(0);
		int y = input.size(1)/h;
		int x = input.size(2)/w;
		if(output==null){
			output = new Tensor(noPlanes, y, x);
		} else {
			// reshape if output was input for linear...
			// TODO check size
			output.reshape(noPlanes, y, x);
		}
		
		output = ModuleOps.spatialavgpool(output, input, w, h, sx, sy, 0, 0, ceil, include_pad);
	}

	@Override
	protected void backward() {	
		if(gradInput == null){
			gradInput = new Tensor(input.dims());
		} 

		ModuleOps.spatialavgpoolGradIn(gradInput, gradOutput, input, w, h, sx, sy, 0, 0, ceil, include_pad);
	}
	
}
