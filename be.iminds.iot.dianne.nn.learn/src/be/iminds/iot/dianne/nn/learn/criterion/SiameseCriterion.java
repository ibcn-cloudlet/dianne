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

public class SiameseCriterion implements Criterion {

	protected Criterion abs;
	protected Criterion mse;
	
	protected BatchConfig config;
	
	protected Tensor grad;
	protected Tensor mean;
	
	protected float alpha = 1f;
	protected float alphadecay = 0.9995f;
	
	public SiameseCriterion(BatchConfig b) {
		this.config = b;
		this.abs = new AbsCriterion(b);
		this.mse = new MSECriterion(b);
	}

	@Override
	public float loss(Tensor output, Tensor target) {
		if(!output.sameDim(grad))
			grad = new Tensor(output.dims());
		
		float error = 0;
		grad.fill(0.0f);
		
		for(int i = 0; i < config.batchSize-1; i++) {
			for(int j = i+1; j < config.batchSize; j++) {
				Tensor o1 = output.narrow(0, i, 1);
				Tensor o2 = output.narrow(0, j, 1);
				Tensor t1 = target.narrow(0, i, 1);
				Tensor t2 = target.narrow(0, j, 1);
				Tensor g1 = grad.narrow(0, i, 1);
				Tensor g2 = grad.narrow(0, j ,1);
				
				float e = abs.loss(o1, o2);
				Tensor g = abs.grad(o1, o2);
				
				if(!t1.equals(t2)) {
					e = -e + 1;
					
					TensorOps.mul(g, g, -1.0f);
				}
				
				error += e;
				
				TensorOps.add(g1, g1, 1.0f, g);
				TensorOps.add(g2, g2, -1.0f, g);
			}
		}
		
		error /= config.batchSize-1;
		TensorOps.div(grad, grad, config.batchSize-1);
		
		if(!output.sameDim(mean)) {
			mean = new Tensor(output.dims());
			mean.fill(0.5f);
		}
		
		float e = mse.loss(output, mean);
		Tensor g = mse.grad(output, mean);
		
		error += alpha*e;
		TensorOps.add(grad, grad, alpha, g);
		
		alpha *= alphadecay;
		
		if(config.batchAverage) {
			error /= config.batchSize;
			TensorOps.div(grad, grad, config.batchSize);
		}
		
		return error;
	}

	@Override
	public Tensor grad(Tensor output, Tensor target) {
		return grad;
	}

}
