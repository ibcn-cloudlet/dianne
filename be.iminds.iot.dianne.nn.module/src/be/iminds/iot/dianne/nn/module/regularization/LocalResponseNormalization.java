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

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.tensor.ModuleOps;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

/**
 * Applies Spatial Local Response Normalization between different feature maps. 
 * 
 * By default, alpha = 0.0001, beta = 0.75 and k = 1
 * 
 * @author tverbele
 *
 */
public class LocalResponseNormalization extends AbstractModule {
	
	private int size;
	private float alpha = 0.0001f;
	private float beta = 0.75f;
	private float k = 1;
	
	private Tensor temp;
	
	public LocalResponseNormalization(int size) {
		super();
		this.size = size;
	}
	
	public LocalResponseNormalization(UUID id, int size) {
		super(id);
		this.size = size;
	}

	public LocalResponseNormalization(int size, float alpha, float beta, float k) {
		super();
		this.size = size;
		this.alpha = alpha;
		this.beta = beta;
		this.k = k;
	}
	
	public LocalResponseNormalization(UUID id, int size, float alpha, float beta, float k) {
		super(id);
		this.size = size;
		this.alpha = alpha;
		this.beta = beta;
		this.k = k;
	}
	
	
	@Override
	protected void forward() {
		// This is the accross channels impl
		System.out.println(input);
		// add zero padding before volumetric avg pool
		int[] dims = input.dims();
		dims[dims.length-3] = dims[dims.length-3] + size-1;
		if(temp == null || !temp.hasDim(dims)) {
			temp = new Tensor(dims);
		}
		temp.fill(0.0f);
		TensorOps.pow(temp.narrow(dims.length-3, (size-1)/2, input.dims()[dims.length-3]), input, 2);
		temp.reshape(1, temp.dims());
		System.out.println(temp);
		output = ModuleOps.volumetricavgpool(output, temp, 1, 1, size, 1, 1, 1);
		System.out.println(output);
		output = TensorOps.mul(output, output, alpha);
		System.out.println(output);
		output = TensorOps.add(output, output, k);
		System.out.println(output);
		output = TensorOps.pow(output, output, beta);
		System.out.println(output);
		output = TensorOps.cdiv(output, input, output);
	}

	@Override
	protected void backward() {
		throw new RuntimeException("Backward not implemented for LNR");
	}

}
