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
 *     Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.learn.criterion;

import be.iminds.iot.dianne.api.nn.learn.Criterion;
import be.iminds.iot.dianne.nn.learn.criterion.CriterionFactory.BatchConfig;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class GaussianCriterion implements Criterion {
	
	protected Tensor meanDiff;
	protected Tensor logStdev;
	
	protected Tensor loss;
	protected Tensor grad;

	protected BatchConfig b;
	
	public GaussianCriterion(BatchConfig b) {
		this.b = b;
	}
	
	@Override
	public Tensor loss(Tensor params, Tensor data) {
		int dim = params.dim()-1;
		int size = params.size(dim)/2;
		
		Tensor mean = params.narrow(dim, 0, size);
		Tensor stdev = params.narrow(dim, size, size);
		
		meanDiff = TensorOps.sub(meanDiff, data, mean);
		
		loss = TensorOps.cdiv(loss, meanDiff, stdev);
		TensorOps.cmul(loss, loss, loss);
		TensorOps.add(loss, loss, (float) Math.log(2*Math.PI));
		TensorOps.div(loss, loss, 2);
		logStdev = TensorOps.log(logStdev, stdev);
		TensorOps.add(loss, loss, logStdev);
		
		return loss;
	}

	@Override
	public Tensor grad(Tensor params, Tensor data) {
		int dim = params.dim()-1;
		int size = params.size(dim)/2;
		
		grad = params.copyInto(grad);
		
		Tensor stdev = params.narrow(dim, size, size);
		Tensor gradMean = grad.narrow(dim, 0, size);
		Tensor gradStdev = grad.narrow(dim, size, size);
		
		TensorOps.cdiv(gradMean, meanDiff, stdev);
		TensorOps.cdiv(gradMean, gradMean, stdev);
		TensorOps.mul(gradMean, gradMean, -1);
		
		gradMean.copyInto(gradStdev);
		TensorOps.cmul(gradStdev, gradStdev, meanDiff);
		TensorOps.add(gradStdev, gradStdev, 1);
		TensorOps.cdiv(gradStdev, gradStdev, stdev);
		
		if(b.batchAverage){
			TensorOps.div(grad, grad, b.batchSize);
		}
			
		return grad;
	}

}
