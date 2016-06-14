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

import be.iminds.iot.dianne.api.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class Convolution extends AbstractTrainableModule {

	protected int noInputPlanes;
	protected int noOutputPlanes;
	protected int kernelWidth = 1;
	protected int kernelHeight = 1;
	protected int kernelDepth = 1;
	protected int strideX = 1;
	protected int strideY = 1;
	protected int strideZ = 1;
	protected int padX = 0;
	protected int padY = 0;
	protected int padZ = 0;

	protected Type type;
	
	// subtensors for weights / bias
	protected Tensor weights;
	protected Tensor deltaWeights;
	protected Tensor bias;
	protected Tensor deltaBias;
	
	// tensors for unfolded data
	protected Tensor temp1 = new Tensor();
	protected Tensor temp2 = new Tensor();

	protected int[] outputDims;
	
	/* Temporal Convolution constructors */
	public Convolution(
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth,
			int stride){
		super(new Tensor(noOutputPlanes*noInputPlanes*kernelWidth+noOutputPlanes));
		type = Type.TEMPORAL;
		init(noInputPlanes, noOutputPlanes, kernelWidth, 1, 1, stride, 1, 1, 0, 0, 0);
		parameters.fill(0.0f);
	}
	
	public Convolution(UUID id,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth,
			int stride){
		super(id, new Tensor(noOutputPlanes*noInputPlanes*kernelWidth+noOutputPlanes));
		type = Type.TEMPORAL;
		init(noInputPlanes, noOutputPlanes, kernelWidth, 1, 1, stride, 1, 1, 0, 0, 0);
		parameters.fill(0.0f);
	}
	
	public Convolution(UUID id, Tensor parameters,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth,
			int stride){
		super(id, parameters);
		type = Type.TEMPORAL;
		init(noInputPlanes, noOutputPlanes, kernelWidth, 1, 1, stride, 1, 1, 0, 0, 0);
	}
	
	/* Spatial Convolution constructors */
	public Convolution(
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight,
			int strideX, int strideY, 
			int padX, int padY){
		super(new Tensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes));
		type = Type.SPATIAL;
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, 1, strideX, strideY, 1, padX, padY, 0);
		parameters.fill(0.0f);
	}
	
	public Convolution(UUID id,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight,
			int strideX, int strideY,
			int padX, int padY){
		super(id, new Tensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes));
		type = Type.SPATIAL;
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, 1, strideX, strideY, 1, padX, padY, 0);
		parameters.fill(0.0f);
	}
	
	public Convolution(UUID id, Tensor parameters,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight,
			int strideX, int strideY, 
			int padX, int padY){
		super(id, parameters);
		type = Type.SPATIAL;
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, 1, strideX, strideY, 1, padX, padY, 0);
	}
	
	/* Volumetric Convolution constructors */
	public Convolution(
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight, int kernelDepth,
			int strideX, int strideY, int strideZ,
			int padX, int padY, int padZ){
		super(new Tensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth+noOutputPlanes));
		type = Type.VOLUMETRIC;
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, kernelDepth, 
				strideX, strideY, strideZ, padX, padY, padZ);
		parameters.fill(0.0f);
	}
	
	public Convolution(UUID id,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight, int kernelDepth,
			int strideX, int strideY, int strideZ,
			int padX, int padY, int padZ){
		super(id, new Tensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth+noOutputPlanes));
		type = Type.VOLUMETRIC;
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, kernelDepth, 
				strideX, strideY, strideZ, padX, padY, padZ);	
		parameters.fill(0.0f);
	}
	
	public Convolution(UUID id, Tensor parameters,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight, int kernelDepth,
			int strideX, int strideY, int strideZ,
			int padX, int padY, int padZ){
		super(id, parameters);
		type  = Type.VOLUMETRIC;
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, kernelDepth, 
				strideX, strideY, strideZ, padX, padY, padZ);
	}
	
	protected void init(int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight, int kernelDepth,
			int strideX, int strideY, int strideZ,
			int padX, int padY, int padZ){
		if(parameters.size()!=(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth+noOutputPlanes)){
			parameters.reshape(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth+noOutputPlanes);
		}
		
		this.noInputPlanes = noInputPlanes;
		this.noOutputPlanes = noOutputPlanes;
		this.kernelWidth = kernelWidth;
		this.kernelHeight = kernelHeight;
		this.kernelDepth = kernelDepth;
		this.strideX = strideX;
		this.strideY = strideY;
		this.strideZ = strideZ;
		this.padX = padX;
		this.padY = padY;
		this.padZ = padZ;
		
		weights = parameters.narrow(0, 0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth);
		bias = parameters.narrow(0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth, noOutputPlanes);
		weights.reshape(noOutputPlanes, noInputPlanes*kernelDepth*kernelHeight*kernelWidth);
	}
	
	public void initDeltaParameters(Tensor deltas){
		if(deltas==null){
			deltaParameters = new Tensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth+noOutputPlanes);
		} else {
			// TODO check size?
			deltaParameters = deltas;
		}
		deltaWeights = deltaParameters.narrow(0, 0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth);
		deltaBias = deltaParameters.narrow(0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight*kernelDepth, noOutputPlanes);
		
		deltaParameters.fill(0.0f);
		deltaWeights.reshape(noOutputPlanes, noInputPlanes*kernelDepth*kernelHeight*kernelWidth);
	}
	
	@Override
	public void randomize(){
		// randomize weights uniform [-std, std] with std = 1/sqrt(kW*kH*noInputPlanes)  [from torch]
		parameters.rand();
		float std = (float) (1f/Math.sqrt(kernelWidth*kernelHeight*kernelDepth*noInputPlanes));
		TensorOps.mul(parameters, parameters, 2*std);
		TensorOps.sub(parameters, parameters, std);
	}
	
	@Override
	protected void forward() {
		// TODO check input planes dim?
		// TODO check kernel sizes
		switch(type){
		case TEMPORAL:
			if(input.dim() == 1){
				input.reshape(input.dims()[0], 1);
			} 
			output = ModuleOps.temporalconvolve(output, input, weights, bias, kernelWidth, strideX, noInputPlanes, noOutputPlanes);
			break;
		case SPATIAL:
			if(input.dim() == 2 && noInputPlanes == 1) {
				input.reshape(1, input.size(0), input.size(1));
			}
			output = ModuleOps.spatialconvolve(output, input, weights, bias, temp1, temp2, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
			break;
		case VOLUMETRIC:
			output = ModuleOps.volumetricconvolve(output, input, weights, bias, temp1, temp2, 
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
			gradInput = ModuleOps.temporalconvolveGradIn(gradInput, gradOutput, weights, input, kernelWidth, strideX);
			break;
		case SPATIAL:
			gradInput = ModuleOps.spatialconvolveGradIn(gradInput, gradOutput, weights, input, temp1, temp2, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
			break;
		case VOLUMETRIC:
			gradInput = ModuleOps.volumetricconvolveGradIn(gradInput, gradOutput, weights, input, temp1, temp2,
					kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
			break;
		}
		
	}

	@Override
	public void accGradParameters() {
		switch(type){
		case TEMPORAL:
			ModuleOps.temporalconvolveAccGrad(deltaWeights, deltaBias, gradOutput, input, kernelWidth, strideX);
			break;
		case SPATIAL:
			ModuleOps.spatialconvolveAccGrad(deltaWeights, deltaBias, gradOutput, input, temp1, temp2, kernelWidth, kernelHeight, strideX, strideY, padX, padY);
			break;
		case VOLUMETRIC:
			ModuleOps.volumetricconvolveAccGrad(deltaWeights, deltaBias, gradOutput, input, temp1, temp2,
					kernelWidth, kernelHeight, kernelDepth, strideX, strideY, strideZ, padX, padY, padZ);
			break;
		}
	}
}
