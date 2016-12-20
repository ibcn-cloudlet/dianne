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
package be.iminds.iot.dianne.nn.module.vae;

import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.Mimo;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;

/**
 * This module interprets the input(s) as a multivariate gaussian of d dimensions.
 * We for now assume a diagonal covariance matrix, meaning if we have a tensor
 * with size 2*d, treating the first d values as means and the next d values as stdevs
 * 
 * If there are two inputs, those will be treated as means and stdevs separately
 * 
 * If there are two outputs, it will output a mean and stdev tensor separately
 * 
 * @author tverbele
 *
 */
public class MultivariateGaussian extends Mimo {

	private int size;
	
	private String meanActivation;
	private String stdevActivation;

	private Tensor meanInput;
	private Tensor stdevInput;
	
	private Tensor t;
	private Tensor mean;
	private Tensor stdev;
	
	private Tensor meanGradOutput;
	private Tensor stdevGradOutput;
	
	private Tensor g;
	private Tensor gradMean;
	private Tensor gradStdev;
	
	private int batchSize = 1;
	
	public MultivariateGaussian(int size) {
		super();
		this.size = size;
		init();
	}

	public MultivariateGaussian(int size, String meanActivation, String stdevActivation) {
		super();
		this.size = size;
		this.meanActivation = meanActivation == null || meanActivation.isEmpty() ? null : meanActivation.toUpperCase();
		this.stdevActivation = stdevActivation == null || stdevActivation.isEmpty() ? null :stdevActivation.toUpperCase();
		init();
	}
	
	public MultivariateGaussian(UUID id, int size) {
		super(id);
		this.size = size;
		init();
	}
	
	public MultivariateGaussian(UUID id, int size, String meanActivation, String stdevActivation) {
		super(id);
		this.size = size;
		this.meanActivation = meanActivation == null || meanActivation.isEmpty() ? null : meanActivation.toUpperCase();
		this.stdevActivation = stdevActivation == null || stdevActivation.isEmpty() ? null :stdevActivation.toUpperCase();
		init();
	}
	
	private void init(){
		if(batchSize > 1)
			t = new Tensor(batchSize, size*2);
		else 
			t = new Tensor(size*2);
		mean = t.narrow(batchSize > 1 ? 1 : 0, 0, size);
		stdev = t.narrow(batchSize > 1 ? 1 : 0, size, size);
		
		if(batchSize > 1)
			g = new Tensor(batchSize,size*2);
		else
			g = new Tensor(size*2);
		gradMean = g.narrow(batchSize > 1 ? 1 : 0, 0, size);
		gradStdev = g.narrow(batchSize > 1 ? 1 : 0, size, size);
	}
	
	@Override
	protected void forward() {
		Tensor i = inputs.get(prevIds[0]);
		int b = i.size()/(2*size);
		if(batchSize != b){
			batchSize = b;
			init();
		}
		
		if(prev.length == 1){
			// single input tensor
			Tensor input = inputs.get(prevIds[0]);
			meanInput = input.narrow(batchSize > 1 ? 1 : 0, 0, size);
			stdevInput = input.narrow(batchSize > 1 ? 1 : 0, size, size);
		} else if(prev.length == 2){
			// separate mean/stdev tensor
			meanInput = inputs.get(prevIds[0]);
			stdevInput = inputs.get(prevIds[1]);
		} else {
			throw new RuntimeException("Invalid number of prevs, should be 1 or 2");
		}
		
		if(meanActivation != null){
			switch(meanActivation){
			case "SIGMOID":
				ModuleOps.sigmoid(mean, meanInput);
				break;
			case "TANH":
				ModuleOps.tanh(mean, meanInput);
				break;
			default:
				throw new RuntimeException("Invalid mean activation: "+meanActivation);
			}
		} else {
			meanInput.copyInto(mean);
		}
		
		if(stdevActivation != null){
			switch(stdevActivation){
			case "SIGMOID":
				ModuleOps.sigmoid(stdev, stdevInput);
				break;
			case "SOFTPLUS":
				ModuleOps.softplus(stdev, stdevInput, 1, 20);
				break;
			default:
				throw new RuntimeException("Invalid stdev activation: "+stdevActivation);
			}
		} else {
			stdevInput.copyInto(stdev);
		}
		
		if(next.length == 1){
			outputs.put(nextIds[0], t);
		} else if(next.length == 2){
			outputs.put(nextIds[0], mean);
			outputs.put(nextIds[1], stdev);
		} else {
			throw new RuntimeException("Invalid number of nexts, should be 1 or 2");
		}
	}

	@Override
	protected void backward() {
		if(next.length == 1){
			Tensor gradOutput = gradOutputs.get(nextIds[0]);
			meanGradOutput = gradOutput.narrow(batchSize > 1 ? 1 : 0, 0, size);
			stdevGradOutput = gradOutput.narrow(batchSize > 1 ? 1 : 0, size, size);
		} else if(next.length == 2){
			meanGradOutput = gradOutputs.get(nextIds[0]);
			stdevGradOutput = gradOutputs.get(nextIds[1]);
		} else {
			throw new RuntimeException("Invalid number of prevs, should be 1 or 2");
		}
		
		if(meanActivation != null){
			switch(meanActivation){
			case "SIGMOID":
				ModuleOps.sigmoidGradIn(gradMean, meanGradOutput, meanInput, mean);
				break;
			case "TANH":
				ModuleOps.tanhGradIn(gradMean, meanGradOutput, meanInput, mean);
				break;
			default:
				throw new RuntimeException("Invalid mean activation: "+meanActivation);
			}
		} else {
			meanGradOutput.copyInto(gradMean);
		}
		
		if(stdevActivation != null){
			switch(stdevActivation){
			case "SIGMOID":
				ModuleOps.sigmoidGradIn(gradStdev, stdevGradOutput, stdevInput, stdev);
				break;
			case "SOFTPLUS":
				ModuleOps.softplusGradIn(gradStdev, stdevGradOutput, stdevInput, stdev, 1, 20);
				break;
			default:
				throw new RuntimeException("Invalid stdev activation: "+stdevActivation);
			}
		} else {
			stdevGradOutput.copyInto(gradStdev);
		}
		
		if(prev.length == 1){
			gradInputs.put(prevIds[0], g);
		} else if(prev.length == 2){
			gradInputs.put(prevIds[0], gradMean);
			gradInputs.put(prevIds[1], gradStdev);
		} else {
			throw new RuntimeException("Invalid number of nexts, should be 1 or 2");
		}
	}

}
