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
	
	protected static final float EPS = 1e-6f;
	
	protected Tensor outStdev;
	protected Tensor tarStdev;
	
	protected Tensor stdevRatio;
	protected Tensor logStdevRatio;
	
	protected Tensor sqTarStdev;
	protected Tensor invOutStdev;

	protected Tensor invTarStdev;

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
		
		// loss = log(s_tar / s_out) + (s_out^2 + (mu_out - mu_tar)^2) / (2 * s_tar^2) - 1/2
		//      = ((s_out / s_tar)^2 + ((mu_out - mu_tar) / s_tar)^2 - log((s_out / s_tar)^2) - 1) / 2
		Tensor outMean = output.narrow(dim, 0, size);
		Tensor tarMean = target.narrow(dim, 0, size);
		outStdev = TensorOps.add(outStdev, output.narrow(dim, size, size), EPS);
		tarStdev = TensorOps.add(tarStdev, target.narrow(dim, size, size), EPS);
		
		l = TensorOps.sub(l, outMean, tarMean);
		l = TensorOps.cdiv(l, l, tarStdev);
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
		
		Tensor outMean = output.narrow(dim, 0, size);
		Tensor tarMean = target.narrow(dim, 0, size);
		outStdev = TensorOps.add(outStdev, output.narrow(dim, size, size), EPS);
		tarStdev = TensorOps.add(tarStdev, target.narrow(dim, size, size), EPS);
		Tensor gradMean = grad.narrow(dim, 0, size);
		Tensor gradStdev = grad.narrow(dim, size, size);
		
		sqTarStdev = TensorOps.cmul(sqTarStdev, tarStdev, tarStdev);
		invOutStdev = TensorOps.pow(invOutStdev, outStdev, -1);
		
		// grad mu_out = (mu_out - mu_tar) / s_tar^2
		TensorOps.sub(gradMean, outMean, tarMean);
		TensorOps.cdiv(gradMean, gradMean, sqTarStdev);
		
		// grad s_out = s_out / s_tar^2 - 1 / s_out
		TensorOps.cdiv(gradStdev, outStdev, sqTarStdev);
		TensorOps.sub(gradStdev, gradStdev, invOutStdev);
		
		if(b.batchAverage){
			TensorOps.div(grad, grad, b.batchSize);
		}
			
		return grad;
	}

	@Override
	public Tensor gradTarget(final Tensor output, final Tensor target){
		int dim = output.dim()-1;
		int size = output.size(dim)/2;
		
		grad = output.copyInto(grad);
		
		Tensor outMean = output.narrow(dim, 0, size);
		Tensor tarMean = target.narrow(dim, 0, size);
		outStdev = TensorOps.add(outStdev, output.narrow(dim, size, size), EPS);
		tarStdev = TensorOps.add(tarStdev, target.narrow(dim, size, size), EPS);
		Tensor gradMean = grad.narrow(dim, 0, size);
		Tensor gradStdev = grad.narrow(dim, size, size);
		
		sqTarStdev = TensorOps.cmul(sqTarStdev, tarStdev, tarStdev);
		
		// grad mu_tar = (mu_tar - mu_out) / s_tar^2
		TensorOps.sub(gradMean, tarMean, outMean);
		TensorOps.cdiv(gradMean, gradMean, sqTarStdev);
		
		// grad s_tar = -s_out^2 / s_tar^3 - (mu_out - mu_tar)^2 / s_tar^3 + 1 / (2 * s_tar)
		//            = (-(s_out^2 + (mu_out - mu_tar)^2) / s_tar^2 + 1/2) / s_tar
		TensorOps.sub(gradStdev, outMean, tarMean);
		TensorOps.cmul(gradStdev, gradStdev, gradStdev);
		
		TensorOps.addcmul(gradStdev, gradStdev, 1, outStdev, outStdev);
		TensorOps.cdiv(gradStdev, gradStdev, sqTarStdev);
		
		TensorOps.mul(gradStdev, gradStdev, -1);
		TensorOps.add(gradStdev, gradStdev, 0.5f);
		TensorOps.cdiv(gradStdev, gradStdev, tarStdev);
		
		if(b.batchAverage){
			TensorOps.div(grad, grad, b.batchSize);
		}
		
		return grad;
	}
}
