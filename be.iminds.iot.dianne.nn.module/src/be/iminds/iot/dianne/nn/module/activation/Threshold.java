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
package be.iminds.iot.dianne.nn.module.activation;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.ModuleOps;

public class Threshold extends AbstractModule {
	
	private final float thresh;
	private final float val;
	
	public Threshold(float thresh, float val) {
		super();
		this.thresh = thresh;
		this.val = val;
	}
	
	public Threshold(UUID id, float thresh, float val) {
		super(id);
		this.thresh = thresh;
		this.val = val;
	}

	@Override
	protected void forward() {
		output = ModuleOps.threshold(output, input, thresh, 0, val);
	}

	@Override
	protected void backward() {
		//gradInput = TensorOps.cmul(gradInput, gradOutput, 
		//		TensorOps.dthresh(gradInput, input, thresh, 0));
		gradInput = ModuleOps.thresholdDin(gradInput, gradOutput, input, thresh, 0);
	}

}
