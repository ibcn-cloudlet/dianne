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
package be.iminds.iot.dianne.api.nn.eval;

import java.util.List;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Result of the evaluation of a Dataset, provides access to the confusion matrix.
 * 
 * @author tverbele
 *
 */
public class Evaluation {

	public long total;
	public float error;
	
	// the actual outputs 
	public List<Tensor> outputs;
	// time to run the full evaluation
	public long evaluationTime;
	// average forward time
	public float forwardTime;
	
	@Override
	public String toString(){
		return "Error: "+error+" ( on "+total+" samples) - Time: "+evaluationTime+" ms.";
	}
	
	/**
	 * @return the total evaluation time
	 */
	public long evaluationTime(){
		return evaluationTime;
	}
	
	/**
	 * @return average time for processing one sample
	 */
	public float forwardTime(){
		return forwardTime;
	}
	
	public List<Tensor> getOutputs(){
		return outputs;
	}
	
	public Tensor getOutput(int index){
		return outputs.get(index);
	}
	
	/**
	 * @return the total number of samples in the evaluation set
	 */
	public long getTotal(){
		return total;
	}
	
	/**
	 * @return error on global dataset
	 */
	public float error() {
		return error;
	}

}
