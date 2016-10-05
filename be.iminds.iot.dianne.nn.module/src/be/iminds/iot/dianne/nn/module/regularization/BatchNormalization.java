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
package be.iminds.iot.dianne.nn.module.regularization;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * In case BatchNormalization is placed after a Linear, size should be the Linear's output size.
 * 
 * In case BatchNormalization is placed after a Convolution, size should be the number of output
 * feature planes of the Convolution.
 * 
 * @author tverbele
 *
 */
public class BatchNormalization extends AbstractTrainableModule{
	
	private int size;
	
	private Tensor weights;
	private Tensor bias;
	
	private Tensor gradWeights;
	private Tensor gradBias;
	
	private Tensor rMean;
	private Tensor rVar;

	private Tensor sMean;
	private Tensor sVar;
	
	private int[] inputDims;
	private int[] bnDims;
	
	public BatchNormalization(int size) {
		super(new Tensor(4*size));
		this.size = size;
		init();
	}
	
	public BatchNormalization(UUID id, int size) {
		super(id, new Tensor(4*size));
		this.size = size;
		init();
	}

	public BatchNormalization(UUID id, Tensor params, int size) {
		super(id, params);
		// TODO check params size = size*4?
		this.size = size;
		init();
	}
	
	private void init() {
		// initialize parameters
		weights = parameters.narrow(0, 0, size);
		bias = parameters.narrow(0, size, size);
		
		// keep running mean and var as parameters to be used in evaluation mode?!
		rMean = parameters.narrow(0, 2*size, size);
		rVar = parameters.narrow(0, 3*size, size);
		
		sMean = new Tensor(size);
		sVar = new Tensor(size);
		
		sMean.fill(0.0f);
		sVar.fill(1.0f);
	}

	@Override
	public void randomize(){
		// do not randomize?!
		weights.rand();
		bias.fill(0.0f);
		
		rMean.fill(0.0f);
		rVar.fill(1.0f);
		
		sMean.fill(0.0f);
		sVar.fill(1.0f);
	}
	
	@Override
	protected void forward() {
		inputDims = input.dims();
		if(inputDims.length == 1){
			// 1D input, single sample
			bnDims = new int[]{1, inputDims[0]};
		} else if(inputDims.length == 2){
			// 1D input, batched
			// no reshape needed
			bnDims = inputDims;
		} else if(inputDims.length == 3){
			// spatial input, single sample
			bnDims = new int[]{1, inputDims[0], inputDims[1]*inputDims[2]};
		} else if(inputDims.length == 4){
			if(inputDims[1] != size){
				// treat as volumetric input, single sample
				// TODO this means that if you have a volumetric single sample with 
				// depth = noFeatures this will be wongly treated as batched spatial input
				bnDims = new int[]{1, inputDims[0], inputDims[1]*inputDims[2]*inputDims[3]};
			} else {
				// spatial input, batched
				bnDims = new int[]{inputDims[0], inputDims[1], inputDims[2]*inputDims[3]};
			}
 		} else if(inputDims.length == 5){
			// volumetric input, batched
			bnDims = new int[]{inputDims[0], inputDims[1], inputDims[2]*inputDims[3]*inputDims[4]};
		}
		
		input.reshape(bnDims);
		output = ModuleOps.batchnorm(output, input, weights, bias, rMean, rVar, sMean, sVar, train);
		output.reshape(inputDims);
		input.reshape(inputDims);
	}

	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters(null);
		}
		
		input.reshape(bnDims);
		gradOutput.reshape(bnDims);
		gradInput = ModuleOps.batchnormGradIn(gradInput, gradOutput, input, weights, rMean, rVar, sMean, sVar, train);
		gradInput.reshape(inputDims);
		gradOutput.reshape(inputDims);
		input.reshape(inputDims);
	}

	@Override
	public void accGradParameters() {
		input.reshape(bnDims);
		gradOutput.reshape(bnDims);
		ModuleOps.batchnormAccGrad(gradWeights, gradBias, gradOutput, input, weights, rMean, rVar, sMean, sVar, train);
		gradOutput.reshape(inputDims);
		input.reshape(inputDims);
	}

	@Override
	public void initDeltaParameters(Tensor deltas) {
		if(deltas==null){
			deltaParameters = new Tensor(size*4);
		} else {
			// TODO check size?
			deltaParameters = deltas;
		}
		deltaParameters.fill(0.0f);  // rMean and rVar are updated during forward passes in train mode...
		
		gradWeights = deltaParameters.narrow(0, 0, size);
		gradBias = deltaParameters.narrow(0, size, size);
	}
}
