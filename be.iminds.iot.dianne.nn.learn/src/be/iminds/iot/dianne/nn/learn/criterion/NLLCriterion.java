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
 * Negative Log Likelihood criterion
 * 
 * Assumes the outputs are log probabilities (i.e. from LogSoftmax output)
 * 
 * @author tverbele
 *
 */
public class NLLCriterion implements Criterion {

	protected Tensor gradInput;
	protected Tensor nll;
	protected Tensor log = null;
	
	public NLLCriterion() {
		this.nll = new Tensor(1);
	}
	
	@Override
	public Tensor error(final Tensor output, final Tensor target) {
		if(output.get()[0] < 0){
			log = null;
			// output comes from LogSoftmax, no log required
			// this should be numerically more stable
			float ll = TensorOps.dot(output, target);
			nll.set(-ll, 0);
		} else {
			// calculate negative log 
			log = TensorOps.log(log, output);
			float ll = TensorOps.dot(log, target);
			nll.set(-ll, 0);
		}
	
		return nll;
	}

	@Override
	public Tensor grad(final Tensor output, final Tensor target) {
		if(log != null){
			gradInput = TensorOps.cdiv(gradInput, target, output);
			gradInput = TensorOps.mul(gradInput, gradInput, -1.0f);
		} else {
			gradInput = TensorOps.mul(gradInput, target, -1.0f);
		}
		return gradInput;
	}
}
