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

import org.junit.Test;

import be.iminds.iot.dianne.nn.module.ModuleTest;
import be.iminds.iot.dianne.tensor.Tensor;

public class TemporalConvolutionTest extends ModuleTest{

	@Test
	public void testTemporalConvolution1() throws InterruptedException {
		int noInputPlanes = 1;
		int noOutputPlanes = 2;
		int kernelWidth = 3;
		int stride = 1;
		Convolution sc = new Convolution(noInputPlanes, noOutputPlanes, 
				kernelWidth, 
				stride);
		
		// temporal has "temporal" dim first and "feature" dim second
		Tensor input = new Tensor(5);
		input.fill(1.0f);
		
		Tensor gradOutput = new Tensor(3, 2);
		gradOutput.fill(1.0f);
		
		Tensor params = new Tensor(noOutputPlanes*(noInputPlanes*kernelWidth+1));
		params.fill(1.0f);
		
		Tensor expOutput = new Tensor(3, 2);
		expOutput.fill(4.0f);
		
		float[] gradInData = new float[]{2.0f, 4.0f, 6.0f, 4.0f, 2.0f};
		Tensor expGradInput = new Tensor(gradInData, 5, 1);
		
		Tensor expDelta = new Tensor(noOutputPlanes*(noInputPlanes*kernelWidth+1));
		expDelta.fill(3.0f);
		
		testModule(sc, params, input, expOutput, gradOutput, expGradInput, expDelta);
	}
	
	@Test
	public void testTemporalConvolution2() throws InterruptedException {
		int noInputPlanes = 2;
		int noOutputPlanes = 2;
		int kernelWidth = 3;
		int stride = 1;
		Convolution sc = new Convolution(noInputPlanes, noOutputPlanes, 
				kernelWidth, 
				stride);
		
		// temporal has "temporal" dim first and "feature" dim second
		Tensor input = new Tensor(5, 2);
		input.fill(1.0f);
		
		Tensor gradOutput = new Tensor(3, 2);
		gradOutput.fill(1.0f);
		
		Tensor params = new Tensor(noOutputPlanes*(noInputPlanes*kernelWidth+1));
		params.fill(1.0f);
		
		Tensor expOutput = new Tensor(3, 2);
		expOutput.fill(7.0f);
		
		float[] gradInData = new float[]{2.0f, 2.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 4.0f, 2.0f, 2.0f};
		Tensor expGradInput = new Tensor(gradInData, 5, 2);
		
		Tensor expDelta = new Tensor(noOutputPlanes*(noInputPlanes*kernelWidth+1));
		expDelta.fill(3.0f);
		
		testModule(sc, params, input, expOutput, gradOutput, expGradInput, expDelta);
	}
	

}
