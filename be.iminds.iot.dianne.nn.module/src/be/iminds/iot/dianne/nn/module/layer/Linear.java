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
import be.iminds.iot.dianne.tensor.TensorFactory;

public class Linear extends AbstractTrainableModule {

	private int inSize;
	private int outSize;
	
	private Tensor weights;
	private Tensor bias;
	
	private Tensor deltaWeights;
	private Tensor deltaBias;
	
	public Linear(TensorFactory factory, int inSize, int outSize){
		super(factory, factory.createTensor(outSize*(inSize+1)));
		init(inSize, outSize);
		parameters.fill(0.0f);
	}
	
	public Linear(TensorFactory factory, UUID id, int inSize, int outSize){
		super(factory, id, factory.createTensor(outSize*(inSize+1)));
		init(inSize, outSize);
		parameters.fill(0.0f);
	}
	
	public Linear(TensorFactory factory, UUID id, Tensor parameters, int inSize, int outSize){
		super(factory, id, parameters);
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
		bias = parameters.narrow(0, outSize*inSize, outSize);
		bias.reshape(outSize);
	}
	
	private void initDeltaParameters(){
		deltaParameters = factory.createTensor(outSize*(inSize+1));
		
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
		factory.getTensorMath().mul(parameters, parameters, 2*std);
		factory.getTensorMath().sub(parameters, parameters, std);		
	}
	
	@Override
	protected void forward() {
		output = factory.getTensorMath().addmv(output, bias, weights, input);
	}

	@Override
	protected void backward() {
		if(deltaParameters==null){
			initDeltaParameters();
		}
		gradInput = factory.getTensorMath().tmv(gradInput, weights, gradOutput);
	}

	@Override
	public void accGradParameters() {
		deltaWeights = factory.getTensorMath().addvv(deltaWeights, deltaWeights, gradOutput, input);
		deltaBias = factory.getTensorMath().add(deltaBias, deltaBias, gradOutput);
	}

}
