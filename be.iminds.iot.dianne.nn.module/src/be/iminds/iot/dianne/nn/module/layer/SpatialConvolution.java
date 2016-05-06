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

public class SpatialConvolution extends AbstractTrainableModule {

	private int noInputPlanes;
	private int noOutputPlanes;
	private int kernelWidth;
	private int kernelHeight;
	private int strideX;
	private int strideY;
	private int padX = 0;
	private int padY = 0;
	
	// subtensors for weights / bias
	Tensor weights;
	Tensor deltaWeights;
	Tensor bias;
	Tensor deltaBias;
	
	public SpatialConvolution(
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight,
			int strideX, int strideY, boolean pad){
		super(new Tensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes));
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, pad);
		parameters.fill(0.0f);
	}
	
	public SpatialConvolution(UUID id,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight,
			int strideX, int strideY, boolean pad){
		super(id, new Tensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes));
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, pad);
		parameters.fill(0.0f);
	}
	
	public SpatialConvolution(UUID id, Tensor parameters,
			int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight,
			int strideX, int strideY, boolean pad){
		super(id, parameters);
		init(noInputPlanes, noOutputPlanes, kernelWidth, kernelHeight, strideX, strideY, pad);
	}
	
	protected void init(int noInputPlanes, int noOutputPlanes, 
			int kernelWidth, int kernelHeight, int strideX, int strideY, boolean pad){
		if(parameters.size()!=(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes)){
			parameters.reshape(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes);
		}
		
		this.noInputPlanes = noInputPlanes;
		this.noOutputPlanes = noOutputPlanes;
		this.kernelWidth = kernelWidth;
		this.kernelHeight = kernelHeight;
		this.strideX = strideX;
		this.strideY = strideY;
		if(pad){
			this.padX = (kernelWidth-1)/2;
			this.padY = (kernelHeight-1)/2;
		}
		
		weights = parameters.narrow(0, 0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight);
		weights.reshape(noOutputPlanes, noInputPlanes, kernelWidth, kernelHeight);
		bias = parameters.narrow(0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight, noOutputPlanes);
	}
	
	public void initDeltaParameters(Tensor deltas){
		if(deltas==null){
			deltaParameters = new Tensor(noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight+noOutputPlanes);
		} else {
			// TODO check size?
			deltaParameters = deltas;
		}
		deltaWeights = deltaParameters.narrow(0, 0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight);
		deltaWeights.reshape(noOutputPlanes, noInputPlanes, kernelWidth, kernelHeight);
		deltaBias = deltaParameters.narrow(0, noOutputPlanes*noInputPlanes*kernelWidth*kernelHeight, noOutputPlanes);
		
		deltaParameters.fill(0.0f);
	}
	
	@Override
	public void randomize(){
		// randomize weights uniform [-std, std] with std = 1/sqrt(kW*kH*noInputPlanes)  [from torch]
		parameters.rand();
		float std = (float) (1f/Math.sqrt(kernelWidth*kernelHeight*noInputPlanes));
		TensorOps.mul(parameters, parameters, 2*std);
		TensorOps.sub(parameters, parameters, std);
	}
	
	@Override
	protected void forward() {
		// TODO check input planes dim?
		// TODO check kernel sizes
		if(input.dim() == 2 && noInputPlanes == 1) {
			input.reshape(1, input.size(0), input.size(1));
		}
		
		output = ModuleOps.spatialconvolve(output, input, weights, bias, strideX, strideY, padX, padY);
	}

	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters(null);
		}
			
		gradOutput.reshape(output.dims());

		gradInput = ModuleOps.spatialconvolveDin(gradInput, gradOutput, weights, strideX, strideY, padX, padY);
	}

	@Override
	public void accGradParameters() {
		deltaWeights = ModuleOps.spatialconvolveDker(deltaWeights, deltaWeights, gradOutput, input, strideX, strideY, padX, padY);
		
		deltaBias = ModuleOps.spatialconvolveDbias(deltaBias, gradOutput);
		// move this to spatialconvolveDbias?
		//for(int i = 0; i < noOutputPlanes; i++)
		//	deltaBias.set(TensorOps.sum(gradOutput.select(0, i)), i);
	}
}
