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
		"dianne.evaluator.category=CLASSIFICATION"})
public class ClassificationEvaluator extends AbstractEvaluator {
	
	protected void evalOutput(int index, Tensor out, Tensor expected){
		if(confusion==null){
			int outputSize = out.size();
			confusion = new Tensor(outputSize, outputSize);
			confusion.fill(0.0f);
		}
		
		int predicted = TensorOps.argmax(out);
		int real = TensorOps.argmax(expected);
		if(real!=predicted)
			error= error+1.0f;
		
		if(trace){
			System.out.println("Sample "+index+" was "+predicted+", should be "+real);
		}
		
		confusion.set(confusion.get(real, predicted)+1, real, predicted);
	}

}

