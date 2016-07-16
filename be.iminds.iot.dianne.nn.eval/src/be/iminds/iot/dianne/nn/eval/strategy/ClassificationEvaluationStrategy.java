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
package be.iminds.iot.dianne.nn.eval.strategy;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;

import be.iminds.iot.dianne.api.nn.eval.ClassificationEvaluation;
import be.iminds.iot.dianne.api.nn.eval.Evaluation;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class ClassificationEvaluationStrategy extends AbstractEvaluationStrategy {
	
	protected Tensor confusion;
	protected int[] rankings;
	
	private int count;
	
	protected void init(Map<String, String> config){
		// reset
		rankings = new int[(int)total];
		confusion = null;
		count = 0;
	}
	
	protected float eval(Tensor output, Tensor target){
		if(confusion==null){
			int confusionSize = output.size();
			if(output.dim() == 2){
				confusionSize = output.size(1);
			}
			confusion = new Tensor(confusionSize, confusionSize);
			confusion.fill(0.0f);
		}
		
		float err = 0;
		if(output.dim() == 2){
			// batch
			for(int i=0;i<output.size(0);i++){
				err += calculateError(output.select(0, i), target.select(0, i));
			}
		} else {
			err += calculateError(output, target); 
		}
		
		return err;
	}
	
	private float calculateError(Tensor out, Tensor expected){
		float error = 0.0f;
		
		int predicted = TensorOps.argmax(out);
		int real = TensorOps.argmax(expected);
		if(real!=predicted)
			error = 1.0f;
		
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
		rankings[count++] = ranking;
		
		return error;
	}
	
	protected Evaluation finish(){
		ClassificationEvaluation eval = new ClassificationEvaluation();
		eval.rankings = rankings;
		eval.confusionMatrix = confusion;
		return eval;
	}
}

