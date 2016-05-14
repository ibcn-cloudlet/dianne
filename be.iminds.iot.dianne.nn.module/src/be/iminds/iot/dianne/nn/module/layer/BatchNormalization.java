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
	
	private int batchSize;
	private int[] inputDims;
	
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
		
		rMean.fill(0.0f);
		rVar.fill(1.0f);
		
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
		batchSize = input.size()/size;
		
		input.reshape(batchSize, size);
		output = ModuleOps.batchnorm(output, input, weights, bias, rMean, rVar, sMean, sVar, true);
		output.reshape(inputDims);
	}

	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters(null);
		}
		
		gradOutput.reshape(batchSize, size);
		gradInput = ModuleOps.batchnormGradIn(gradInput, gradOutput, input, weights, rMean, rVar, sMean, sVar, true);
		gradInput.reshape(inputDims);
	}

	@Override
	public void accGradParameters() {
		ModuleOps.batchnormAccGrad(gradWeights, gradBias, gradOutput, input, rMean, rVar, sMean, sVar, true);
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
