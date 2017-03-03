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
package be.iminds.iot.dianne.api.dataset;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * A helper class for representing raw dataset data
 * 
 * This can be used to tansfer dataset data from a remote dataset 
 * without creating precious native tensors all the time.
 * 
 * @author tverbele
 *
 */
public class RawBatch {
	
	public int[] inputDims;
	public float[] input;
	
	public int[] targetDims;
	public float[] target;
	
	public RawBatch(int[] inputDims, float[] input, int[] targetDims, float[] target){
		this.inputDims = inputDims;
		this.input = input;
		this.targetDims = targetDims;
		this.target = target;
	}
	
	public Batch copyInto(Batch b){
		if(b == null){
			return new Batch(new Tensor(input, inputDims), new Tensor(target, targetDims));
		} else {
			b.input.set(input);
			b.target.set(target);
			return b;
		}
	}
	
}
