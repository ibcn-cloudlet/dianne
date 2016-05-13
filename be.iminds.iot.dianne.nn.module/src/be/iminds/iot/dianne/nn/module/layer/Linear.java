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
package be.iminds.iot.dianne.nn.module.layer;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractTrainableModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class Linear extends AbstractTrainableModule {

	private int inSize;
	private int outSize;
	
	private Tensor weights;
	private Tensor weightsT;
	private Tensor bias;
	
	private Tensor batchedBias = new Tensor();
	
	private Tensor deltaWeights;
	private Tensor deltaBias;

	// keep latest input dimensions
	private int[] inputDims;
	
	public Linear(int inSize, int outSize){
		super(new Tensor(outSize*(inSize+1)));
		init(inSize, outSize);
		parameters.fill(0.0f);
	}
	
	public Linear(UUID id, int inSize, int outSize){
		super(id, new Tensor(outSize*(inSize+1)));
		init(inSize, outSize);
		parameters.fill(0.0f);
	}
	
	public Linear(UUID id, Tensor parameters, int inSize, int outSize){
		super(id, parameters);
		init(inSize, outSize);
	}
	
	private void init(int inSize, int outSize){
		this.inSize = inSize;
		this.outSize = outSize;
		
		if(parameters.size()!=outSize*(inSize+1)){
			parameters.reshape(outSize*(inSize+1));
		}
		
		weights = parameters.narrow(0, 0, outSize*inSize);
		weights.reshape(outSize, inSize);
		weightsT = weights.transpose(null, 0, 1);
		bias = parameters.narrow(0, outSize*inSize, outSize);
		bias.reshape(outSize);
	}
	
	public void initDeltaParameters(Tensor deltas){
		if(deltas==null){
			deltaParameters = new Tensor(outSize*(inSize+1));
		} else {
			// TODO check size?
			deltaParameters = deltas;
		}
		
		deltaWeights = deltaParameters.narrow(0, 0, outSize*inSize);
		deltaWeights.reshape(outSize, inSize);
		deltaBias = deltaParameters.narrow(0, outSize*inSize, outSize);
		deltaBias.reshape(outSize);
		
		deltaParameters.fill(0.0f);
	}
	
	@Override 
	public void randomize(){
		// randomize weights uniform [-std, std] with std = 1/sqrt(noInputs)  [from torch]
		parameters.rand();
		float std = (float) (1f/Math.sqrt(inSize));
		TensorOps.mul(parameters, parameters, 2*std);
		TensorOps.sub(parameters, parameters, std);		
	}
	
	@Override
	protected void forward() {
		inputDims = input.dims();
		if(inputDims.length % 2 == 1){
			// 1 or 3 treat as vector
			if(inputDims.length == 3){
				input.reshape(inputDims[0]*inputDims[1]*inputDims[2]);
			}
			output = TensorOps.addmv(output, bias, weights, input);

		} else {
			// 2 or 4 , treat as batched input
			if(inputDims.length == 4) { 
				input.reshape(inputDims[0], inputDims[1]*inputDims[2]*inputDims[3]);
			}
			int batchSize = input.size(0);
			batchedBias.reshape(batchSize, outSize);
			for(int i=0;i<batchSize;i++){
				bias.copyInto(batchedBias.select(0, i));
			}
			output = TensorOps.addmm(output, batchedBias, input, weightsT);
		}
	}

	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters(null);
		}
		if(inputDims.length % 2 == 1){
			gradInput = TensorOps.mv(gradInput, weightsT, gradOutput);
		} else {
			gradInput = TensorOps.mm(gradInput, gradOutput, weights);
		}
		gradInput.reshape(inputDims);
	}

	@Override
	public void accGradParameters() {
		if(inputDims.length  % 2 == 1){
			deltaWeights = TensorOps.addvv(deltaWeights, deltaWeights, gradOutput, input);
			deltaBias = TensorOps.add(deltaBias, deltaBias, gradOutput);
		} else {
			int batchSize = input.size(0);
			for(int i = 0; i< batchSize; i++){
				deltaWeights = TensorOps.addvv(deltaWeights, deltaWeights, gradOutput.select(0, i), input.select(0, i));
				deltaBias = TensorOps.add(deltaBias, deltaBias, gradOutput.select(0, i));
			}
		}
	}

}
