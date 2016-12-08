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
 * Pseudo-Huber criterion
 * 
 * @author smbohez
 *
 */
public class PseudoHuberCriterion implements Criterion {

	protected Tensor loss = new Tensor(1);
	protected Tensor grad;

	protected Tensor diff;
	protected Tensor sqrt;
	
	protected BatchConfig b;
	
	public PseudoHuberCriterion(BatchConfig b) {
		this.b = b;
	}
	
	@Override
	public Tensor loss(final Tensor output, final Tensor target) {
		diff = TensorOps.sub(diff, output, target);
		sqrt = TensorOps.cmul(sqrt, diff, diff);
		TensorOps.add(sqrt, sqrt, 1);
		TensorOps.sqrt(sqrt, sqrt);
		
		if(b.batchSize > 1){
			loss.reshape(b.batchSize);
			int div = output.size() / b.batchSize;
			for(int i=0;i<b.batchSize;i++){
				Tensor s = sqrt.select(0, i);
				loss.set((TensorOps.sum(s) - s.size())/div, i);
			}
		} else {
			loss.set((TensorOps.sum(sqrt) - sqrt.size()), 0);
		}
		
		return loss;
	}

	@Override
	public Tensor grad(final Tensor output, final Tensor target) {
		int div = output.size() / b.batchSize;
		grad = TensorOps.cdiv(grad, diff, sqrt);
		TensorOps.mul(grad, grad, 1.0f / div);
		
		if(b.batchAverage){
			TensorOps.div(grad, grad, b.batchSize);
		}
		
		return grad;
	}
}
