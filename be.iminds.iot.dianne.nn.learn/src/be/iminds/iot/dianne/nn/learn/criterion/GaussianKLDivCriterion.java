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

public class GaussianKLDivCriterion implements Criterion {
	
	protected Tensor meanDiff;
	protected Tensor stdevRatio;
	protected Tensor logStdevRatio;
	
	protected Tensor sqTarStdev;
	protected Tensor invOutStdev;
	
	protected Tensor l;
	
	protected Tensor loss = new Tensor(1);
	protected Tensor grad;

	protected BatchConfig b;
	
	public GaussianKLDivCriterion(BatchConfig b) {
		this.b = b;
	}
	
	@Override
	public Tensor loss(Tensor output, Tensor target) {
		int dim = output.dim()-1;
		int size = output.size(dim)/2;
		
		Tensor outMean = output.narrow(dim, 0, size);
		Tensor outStdev = output.narrow(dim, size, size);
		Tensor tarMean = target.narrow(dim, 0, size);
		Tensor tarStdev = target.narrow(dim, size, size);
		
		meanDiff = TensorOps.sub(meanDiff, tarMean, outMean);
		l = TensorOps.cdiv(l, meanDiff, tarStdev);
		TensorOps.cmul(l, l, l);
		
		stdevRatio = TensorOps.cdiv(stdevRatio, outStdev, tarStdev);
		TensorOps.cmul(stdevRatio, stdevRatio, stdevRatio);
		TensorOps.add(l, l, stdevRatio);
		
		logStdevRatio = TensorOps.log(logStdevRatio, stdevRatio);
		TensorOps.sub(l, l, logStdevRatio);
		
		TensorOps.sub(l, l, 1);
		TensorOps.div(l, l, 2);
		
		if(b.batchSize > 1){
			loss.reshape(b.batchSize);
			for(int i=0;i<b.batchSize;i++){
				loss.set(TensorOps.sum(l.select(0, i)), i);
			}
		} else {
			loss.set(TensorOps.sum(l), 0);
		}
		
		return loss;
	}

	@Override
	public Tensor grad(Tensor output, Tensor target) {
		int dim = output.dim()-1;
		int size = output.size(dim)/2;
		
		grad = output.copyInto(grad);
		
		Tensor outStdev = output.narrow(dim, size, size);
		Tensor tarMean = target.narrow(dim, 0, size);
		Tensor tarStdev = target.narrow(dim, size, size);
		Tensor gradMean = grad.narrow(dim, 0, size);
		Tensor gradStdev = grad.narrow(dim, size, size);
		
		sqTarStdev = TensorOps.cmul(sqTarStdev, tarStdev, tarStdev);
		invOutStdev = TensorOps.pow(invOutStdev, outStdev, -1);
		
		TensorOps.sub(gradMean, gradMean, tarMean);
		TensorOps.cdiv(gradMean, gradMean, sqTarStdev);
		
		TensorOps.cdiv(gradStdev, gradStdev, sqTarStdev);
		TensorOps.sub(gradStdev, gradStdev, invOutStdev);
		
		if(b.batchAverage){
			TensorOps.div(grad, grad, b.batchSize);
		}
			
		return grad;
	}

}
