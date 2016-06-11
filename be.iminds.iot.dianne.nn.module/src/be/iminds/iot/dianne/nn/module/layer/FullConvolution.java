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
 *     Tim Verbelen, Steven Bohez, Elias De Coninck
 *******************************************************************************/
package be.iminds.iot.dianne.nn.module.layer;

import java.util.UUID;

import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;

public class FullConvolution extends Convolution {
	
	// Temporal not supported (yet)?
	
	/* Spatial FullConvolution constructors */
	public FullConvolution(
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight,
			int strideX, int strideY, 
			int padX, int padY){
		super(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, 
				strideX, strideY, padX, padY);
	}
	
	public FullConvolution(UUID id,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight,
			int strideX, int strideY,
			int padX, int padY){
		super(id, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight,
				strideX, strideY, padX, padY);
	}
	
	public FullConvolution(UUID id, Tensor parameters,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight,
			int strideX, int strideY, 
			int padX, int padY){
		super(id, parameters, noInputPlanes, noOutputPlanes, kernelWidth,
				kernelHeight, strideX, strideY, padX, padY);
	}
	
	/* Volumetric Convolution constructors */
	public FullConvolution(
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight, int kernelDepth,
			int strideX, int strideY, int strideZ,
			int padX, int padY, int padZ){
		super(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight,
				kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
	}
	
	public FullConvolution(UUID id,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight, int kernelDepth,
			int strideX, int strideY, int strideZ,
			int padX, int padY, int padZ){
		super(id, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight,
				kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
	}
	
	public FullConvolution(UUID id, Tensor parameters,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight, int kernelDepth,
			int strideX, int strideY, int strideZ,
			int padX, int padY, int padZ){
		super(id, parameters, noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight,
				kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
	}
	

	// full convolution expects weights shaped with noInputplanes as first dim
	@Override
	protected void init(int noInputPlanes, int noOutputPlanes, int kernelWidth, int kernelHeight, int kernelDepth,
			int strideX, int strideY, int strideZ, int padX, int padY, int padZ) {
		super.init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
		
		switch(type){
		case TEMPORAL:
			weights.reshape(noInputPlanes, noOutputPlanes, kernelWidth);
			break;
		case SPATIAL:
			weights.reshape(noInputPlanes, noOutputPlanes, kernelHeight, kernelWidth);
			break;
		case VOLUMETRIC:
			weights.reshape(noInputPlanes, noOutputPlanes, kernelDepth, kernelHeight, kernelWidth);
		}
	}
	
	@Override
	public void initDeltaParameters(Tensor deltas) {
		super.initDeltaParameters(deltas);
		
		switch(type){
		case TEMPORAL:
			deltaWeights.reshape(noInputPlanes, noOutputPlanes, kernelWidth);
			break;
		case SPATIAL:
			deltaWeights.reshape(noInputPlanes, noOutputPlanes, kernelHeight, kernelWidth);
			break;
		case VOLUMETRIC:
			deltaWeights.reshape(noInputPlanes, noOutputPlanes, kernelDepth, kernelHeight, kernelWidth);
		}
	}
	
	@Override
	protected void forward() {
		switch(type){
		case TEMPORAL:
			// TODO implement temporal using spatial variant?
			throw new UnsupportedOperationException("Temporal full convolution not supported");
		case SPATIAL:
			if(input.dim() == 2 && noInputPlanes == 1) {
				input.reshape(1, input.size(0), input.size(1));
			}
			output = ModuleOps.spatialfullconvolve(output, input, weights, bias, temp1, temp2, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
			break;
		case VOLUMETRIC:
			output = ModuleOps.volumetricfullconvolve(output, input, weights, bias, temp1, temp2, 
					kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
			break;
		}
		
		outputDims = output.dims();
	}

	
	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters(null);
		}
		
		gradOutput.reshape(outputDims);
		switch(type){
		case TEMPORAL:
			// TODO implement temporal using spatial variant?
			throw new UnsupportedOperationException("Temporal full convolution not supported");
		case SPATIAL:
			gradInput = ModuleOps.spatialfullconvolveGradIn(gradInput, gradOutput, weights, input, temp1, temp2, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
			break;
		case VOLUMETRIC:
			gradInput = ModuleOps.volumetricfullconvolveGradIn(gradInput, gradOutput, weights, input, temp1, temp2,
					kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
			break;
		}
		
	}

	@Override
	public void accGradParameters() {
		switch(type){
		case TEMPORAL:
			// TODO implement temporal using spatial variant?
			throw new UnsupportedOperationException("Temporal full convolution not supported");
		case SPATIAL:
			ModuleOps.spatialfullconvolveAccGrad(deltaWeights, deltaBias, gradOutput, input, temp1, temp2, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
			break;
		case VOLUMETRIC:
			ModuleOps.volumetricfullconvolveAccGrad(deltaWeights, deltaBias, gradOutput, input, temp1, temp2,
					kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
			break;
		}
	}
}
