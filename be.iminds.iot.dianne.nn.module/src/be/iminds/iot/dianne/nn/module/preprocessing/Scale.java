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
package be.iminds.iot.dianne.nn.module.preprocessing;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class Scale extends AbstractModule {

	// dims of the scaled tensor
	private int[] targetDims;
	private float[] scaleFactors = null;
	
	private int[] inputDims;
	private int batchSize = 0;
	
	public Scale(final int... dims){
		super();
		this.targetDims = dims;
	}
	
	public Scale(UUID id, final int... dims){
		super(id);
		this.targetDims = dims;
	}
	
	public Scale(final float... factors){
		super();
		this.scaleFactors = factors;
	}

	public Scale(UUID id, final float... factors){
		super(id);
		this.scaleFactors = factors;
	}

	@Override
	protected void forward() {
		inputDims = input.dims();
	
		if(scaleFactors != null){
			targetDims = new int[scaleFactors.length];
			for(int i=0;i<scaleFactors.length;i++){
				targetDims[targetDims.length-1-i] = (int)(inputDims[inputDims.length-1-i]*scaleFactors[scaleFactors.length-1-i]);
			}
		}
			
		if(inputDims.length == targetDims.length+1){
			batchSize = inputDims[0];
		} else {
			batchSize = 0;
		}
		
		// TODO handle batchSize in native code?
		if(batchSize > 0 ){
			if(output == null){
				output = new Tensor(batchSize, targetDims);
		}
	
		for(int i=0;i<batchSize;i++){
			TensorOps.scale2D(output.select(0, i), input.select(0, i), targetDims);
		}
		} else {
			output = TensorOps.scale2D(output, input, targetDims);
		}
	}
	
	@Override
	protected void backward() {
		// backward: scale the gradOutput back?
		// TODO handle batchSize in native code?
		if(batchSize > 0 ){
			if(gradInput == null){
				gradInput = new Tensor(inputDims);
			}

			int[] scaleDims = new int[inputDims.length-1];
			for(int k=0;k<scaleDims.length;k++){
				scaleDims[scaleDims.length-1-k] = inputDims[inputDims.length-1-k];
			}
	
			for(int i=0;i<batchSize;i++){
				TensorOps.scale2D(gradInput.select(0, i), gradOutput.select(0, i), scaleDims);
			}
		} else {
			gradInput = TensorOps.scale2D(gradInput, gradOutput, inputDims);
		}
	}

}
