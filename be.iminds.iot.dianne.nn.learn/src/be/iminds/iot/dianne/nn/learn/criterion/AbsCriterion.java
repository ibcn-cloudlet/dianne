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
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.BatchConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Absolute difference error criterion
 * 
 * @author smbohez
 *
 */
public class AbsCriterion implements Criterion {
	
	protected Tensor diff;
	protected Tensor absdiff;
	protected Tensor loss;
	protected Tensor grad;
	
	protected BatchConfig b;
	
	public AbsCriterion(BatchConfig b) {
		this.b = b;
		this.loss = new Tensor(1);
	}
	
	@Override
	public Tensor loss(final Tensor output, final Tensor target) {
		diff = TensorOps.sub(diff, output, target);
		absdiff = TensorOps.abs(absdiff, diff);
		loss.set(TensorOps.sum(absdiff) / (output.dim() == 2 ? output.size(1) : output.size(0)), 0);
		
		if(b.batchSize > 1 && b.batchAverage){
			TensorOps.div(loss, loss, b.batchSize);
		}
		
		return loss;
	}

	@Override
	public Tensor grad(final Tensor output, final Tensor target) {
		grad = TensorOps.sign(grad, diff);
		grad = TensorOps.mul(grad, grad, 1.0f / (output.dim() == 2 ? output.size(1) : output.size(0)));
		
		if(b.batchSize > 1 && b.batchAverage){
			TensorOps.div(grad, grad, b.batchSize);
		}
		
		return grad;
	}
}
