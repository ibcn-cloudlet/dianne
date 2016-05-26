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
package be.iminds.iot.dianne.nn.eval;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(
		service={Evaluator.class},
		property={"aiolos.unique=true",
		"dianne.evaluator.category=REGRESSION"})
public class RegressionEvaluator extends AbstractEvaluator {
	
	private Tensor err;
	private Tensor sqerr;
	
	protected void evalOutput(int index, Tensor out, Tensor expected){
		
		// for now fixed MSE error ... TODO allow different metrics (use Criterion?)
		err = TensorOps.sub(err, out, expected);
		sqerr = TensorOps.cmul(sqerr, err, err);
		error = error + TensorOps.sum(sqerr)*0.5f;
		
		if(trace){
			System.out.println("Sample "+index+" was "+expected+", should be "+out);
		}
	}

}

