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
 * Negative Log Likelihood criterion
 * 
 * Assumes the outputs are log probabilities (i.e. from LogSoftmax output)
 * 
 * @author tverbele
 *
 */
public class NLLCriterion implements Criterion {

	protected Tensor loss = new Tensor(1);
	protected Tensor grad;
	protected Tensor log;
	
	protected BatchConfig b;
	
	public NLLCriterion(BatchConfig b) {
		this.b = b;
	}
	
	@Override
	public Tensor loss(final Tensor output, final Tensor target) {
		if(output.get()[0] <= 0){
			log = null;
			// output comes from LogSoftmax, no log required
			// this should be numerically more stable
		} else {
			// calculate log 
			log = TensorOps.log(log, output);
		}
		
		if(b.batchSize > 1){
			loss.reshape(b.batchSize);
			for(int i=0;i<b.batchSize;i++){
				loss.set(-TensorOps.dot(log==null ? output.select(0, i) : log.select(0, i) , target.select(0,  i)), i);
			}
		} else {
			loss.set(-TensorOps.dot(log==null ? output : log , target), 0);
		}
	
		return loss;
	}

	@Override
	public Tensor grad(final Tensor output, final Tensor target) {
		if(log != null){
			grad = TensorOps.cdiv(grad, target, output);
			grad = TensorOps.mul(grad, grad, -1.0f);
		} else {
			grad = TensorOps.mul(grad, target, -1.0f);
		}
		
		if(b.batchAverage){
			TensorOps.div(grad, grad, b.batchSize);
		}
		
		return grad;
	}
}
