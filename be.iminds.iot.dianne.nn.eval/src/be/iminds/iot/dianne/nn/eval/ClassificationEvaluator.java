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

import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;

import org.osgi.service.component.annotations.Component;

import be.iminds.iot.dianne.api.nn.eval.ClassificationEvaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluator;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

@Component(
		service={Evaluator.class},
		property={"aiolos.unique=true",
		"dianne.evaluator.category=CLASSIFICATION"})
public class ClassificationEvaluator extends AbstractEvaluator {
	
	protected Tensor confusion;
	protected int[] rankings;
	
	protected void init(Map<String, String> config){
		// reset
		rankings = new int[total];
		confusion = null;
	}
	
	protected float evalOutput(int index, Tensor out, Tensor expected){
		if(confusion==null){
			int outputSize = out.size();
			confusion = new Tensor(outputSize, outputSize);
			confusion.fill(0.0f);
		}
		
		float error = 0.0f;
		
		int predicted = TensorOps.argmax(out);
		int real = TensorOps.argmax(expected);
		if(real!=predicted)
			error = 1.0f;
		
		if(this.config.trace){
			System.out.println("Sample "+index+" was "+predicted+", should be "+real);
		}
		
		confusion.set(confusion.get(real, predicted)+1, real, predicted);
		
		Integer[] indices = new Integer[out.size()];
		for(int i=0;i<out.size();i++){
			indices[i] = i;
		}
		Arrays.sort(indices, new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				float v1 = out.get(o1);
				float v2 = out.get(o2);
				// inverse order to have large->small order
				return v1 > v2 ? -1 : (v1 < v2 ? 1 : 0);
			}
		});
		int ranking = 0;
		for(int i=0;i<indices.length;i++){
			if(indices[i] == real){
				break;
			}
			ranking++;
		}
		rankings[index] = ranking;
		
		return error;
	}
	
	protected Evaluation finish(){
		ClassificationEvaluation eval = new ClassificationEvaluation();
		eval.rankings = rankings;
		eval.confusionMatrix = confusion;
		return eval;
	}
}

