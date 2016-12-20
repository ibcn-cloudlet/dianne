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

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * This module interprets the input as a multivariate gaussian of d dimensions,
 * and samples a value from this distribution.
 * 
 * @author tverbele
 *
 */
public class GaussianSampler extends AbstractModule {

	private int size;
	
	private Tensor mean;
	private Tensor stdev;
	
	private Tensor random;
	
	private Tensor gradMean;
	private Tensor gradStdev;
	
	private int batchSize = 1;
	
	public GaussianSampler(int size) {
		super();
		this.size = size;
		init();
	}

	public GaussianSampler(UUID id, int size) {
		super(id);
		this.size = size;
		init();
	}
	
	private void init(){
		if(batchSize > 1){
			output = new Tensor(batchSize, size);
			random = new Tensor(batchSize, size);
		} else { 
			output = new Tensor(size);
			random = new Tensor(size);
		}
		
		if(batchSize > 1)
			gradInput = new Tensor(batchSize,size*2);
		else
			gradInput = new Tensor(size*2);
		gradMean = gradInput.narrow(batchSize > 1 ? 1 : 0, 0, size);
		gradStdev = gradInput.narrow(batchSize > 1 ? 1 : 0, size, size);
	}
	
	@Override
	protected void forward() {
		int b = input.size()/(2*size);
		if(batchSize != b){
			batchSize = b;
			init();
		}
		
		mean = input.narrow(batchSize > 1 ? 1 : 0, 0, size);
		stdev = input.narrow(batchSize > 1 ? 1 : 0, size, size);
		
		// sample output
		random.randn();
		TensorOps.cmul(output, random, stdev);
		TensorOps.add(output, output, mean);
	}

	@Override
	protected void backward() {
		gradInput.fill(0.0f);

		// calculate gradInput
		TensorOps.add(gradMean, gradMean, gradOutput);
		TensorOps.addcmul(gradStdev, gradStdev, 1, gradOutput, random);
	}

}
