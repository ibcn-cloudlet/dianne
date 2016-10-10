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
package be.iminds.iot.dianne.nn.learn.criterion;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Mean squared error criterion
 * 
 * @author tverbele
 *
 */
public class MSECriterion implements Criterion {

	private Tensor diff;
	private Tensor error;
	private Tensor grad;
	
	private float div = 1;
	
	public MSECriterion() {
		this.error = new Tensor(1);
	}
	
	@Override
	public Tensor loss(final Tensor output, final Tensor target) {
		diff = TensorOps.sub(diff, output, target);
		
		// to determine if it is batch or not ... use following rule of thumb
		// 1d = no batch, 2d = batched 1d, 3d = no batch image, 4d = batched image, 5d = batched volumetric
		int[] dims = output.dims();
		int d = dims.length;
		if(d == 2){
			div = dims[1];
		} else if(d == 4){
			div = dims[1]*dims[2]*dims[3];
		} else if(d == 5){
			div = dims[1]*dims[2]*dims[3]*dims[4];
		} else {
			div = output.size();
		}
		error.set(TensorOps.dot(diff, diff) / div, 0);		
		return error;
	}

	@Override
	public Tensor grad(final Tensor output, final Tensor target) {
		grad = TensorOps.mul(grad, diff, 2.0f / div);
		return grad;
	}

}
