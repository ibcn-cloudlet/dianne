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

public class VolumetricConvolutionTest extends ModuleTest{

	@Test
	public void testVolumetricConvolution1() throws InterruptedException {
		int noInputPlanes = 2;
		int noOutputPlanes = 2;
		int kernelWidth = 3;
		int kernelHeight = 3;
		int kernelDepth = 3;
		int stride = 1;
		int padding = 0;
		Convolution sc = new Convolution(noInputPlanes, noOutputPlanes, 
				kernelWidth, kernelHeight, kernelDepth, 
				stride, stride, stride, 
				padding, padding, padding);
		
		Tensor input = new Tensor(2,3,5,5);
		input.fill(1.0f);
		
		Tensor gradOutput = new Tensor(2,1,3,3);
		gradOutput.fill(1.0f);
		
		Tensor params = new Tensor(noOutputPlanes, (noInputPlanes*kernelDepth*kernelHeight*kernelWidth+1));
		params.fill(1.0f);
		
		Tensor expOutput = new Tensor(2, 1, 3, 3);
		expOutput.fill(55f);
		
		float[] gradInputData = new float[]{
				2.0f, 4.0f, 6.0f, 4.0f, 2.0f, 4.0f, 8.0f, 12.0f, 8.0f, 4.0f,
				6.0f, 12.0f, 18.0f, 12.0f, 6.0f, 4.0f, 8.0f, 12.0f, 8.0f, 4.0f,
				2.0f, 4.0f, 6.0f, 4.0f, 2.0f, 2.0f, 4.0f, 6.0f, 4.0f, 2.0f, 4.0f,
				8.0f, 12.0f, 8.0f, 4.0f, 6.0f, 12.0f, 18.0f, 12.0f, 6.0f, 4.0f,
				8.0f, 12.0f, 8.0f, 4.0f, 2.0f, 4.0f, 6.0f, 4.0f, 2.0f, 2.0f, 4.0f,
				6.0f, 4.0f, 2.0f, 4.0f, 8.0f, 12.0f, 8.0f, 4.0f, 6.0f, 12.0f, 18.0f,
				12.0f, 6.0f, 4.0f, 8.0f, 12.0f, 8.0f, 4.0f, 2.0f, 4.0f, 6.0f, 4.0f,
				2.0f, 2.0f, 4.0f, 6.0f, 4.0f, 2.0f, 4.0f, 8.0f, 12.0f, 8.0f, 4.0f,
				6.0f, 12.0f, 18.0f, 12.0f, 6.0f, 4.0f, 8.0f, 12.0f, 8.0f, 4.0f,
				2.0f, 4.0f, 6.0f, 4.0f, 2.0f, 2.0f, 4.0f, 6.0f, 4.0f, 2.0f, 4.0f,
				8.0f, 12.0f, 8.0f, 4.0f, 6.0f, 12.0f, 18.0f, 12.0f, 6.0f, 4.0f,
				8.0f, 12.0f, 8.0f, 4.0f, 2.0f, 4.0f, 6.0f, 4.0f, 2.0f, 2.0f, 4.0f,
				6.0f, 4.0f, 2.0f, 4.0f, 8.0f, 12.0f, 8.0f, 4.0f, 6.0f, 12.0f, 18.0f,
				12.0f, 6.0f, 4.0f, 8.0f, 12.0f, 8.0f, 4.0f, 2.0f, 4.0f, 6.0f, 4.0f, 2.0f};
		Tensor expGradInput = new Tensor(gradInputData, 2, 3, 5, 5);
		
		Tensor expDelta = new Tensor(noOutputPlanes*(noInputPlanes*kernelDepth*kernelHeight*kernelWidth+1));
		expDelta.fill(9.0f);
		
		testModule(sc, params, input, expOutput, gradOutput, expGradInput, expDelta);
	}
	
	@Test
	public void testVolumetricConvolution2() throws InterruptedException {
		int noInputPlanes = 2;
		int noOutputPlanes = 2;
		int kernelWidth = 3;
		int kernelHeight = 3;
		int kernelDepth = 3;
		int stride = 1;
		int padding = 1;
		Convolution sc = new Convolution(noInputPlanes, noOutputPlanes, 
				kernelWidth, kernelHeight, kernelDepth, 
				stride, stride, stride, 
				padding, padding, padding);
		
		Tensor input = new Tensor(2,3,5,5);
		input.fill(1.0f);
		
		Tensor gradOutput = new Tensor(2,3,5,5);
		gradOutput.fill(1.0f);
		
		Tensor params = new Tensor(noOutputPlanes, (noInputPlanes*kernelDepth*kernelHeight*kernelWidth+1));
		params.fill(1.0f);
		
		float[] expOutputData = new float[]{
				17.0f, 25.0f, 25.0f, 25.0f, 17.0f, 25.0f, 37.0f, 37.0f, 37.0f, 
				25.0f, 25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 25.0f, 37.0f, 37.0f, 
				37.0f, 25.0f, 17.0f, 25.0f, 25.0f, 25.0f, 17.0f, 25.0f, 37.0f, 
				37.0f, 37.0f, 25.0f, 37.0f, 55.0f, 55.0f, 55.0f, 37.0f, 37.0f, 
				55.0f, 55.0f, 55.0f, 37.0f, 37.0f, 55.0f, 55.0f, 55.0f, 37.0f, 
				25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 17.0f, 25.0f, 25.0f, 25.0f, 
				17.0f, 25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 25.0f, 37.0f, 37.0f, 
				37.0f, 25.0f, 25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 17.0f, 25.0f, 
				25.0f, 25.0f, 17.0f, 17.0f, 25.0f, 25.0f, 25.0f, 17.0f, 25.0f, 
				37.0f, 37.0f, 37.0f, 25.0f, 25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 
				25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 17.0f, 25.0f, 25.0f, 25.0f, 
				17.0f, 25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 37.0f, 55.0f, 55.0f, 
				55.0f, 37.0f, 37.0f, 55.0f, 55.0f, 55.0f, 37.0f, 37.0f, 55.0f, 
				55.0f, 55.0f, 37.0f, 25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 17.0f, 
				25.0f, 25.0f, 25.0f, 17.0f, 25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 
				25.0f, 37.0f, 37.0f, 37.0f, 25.0f, 25.0f, 37.0f, 37.0f, 37.0f, 
				25.0f, 17.0f, 25.0f, 25.0f, 25.0f, 17.0f
		};
		Tensor expOutput = new Tensor(expOutputData, 2, 3, 5, 5);
		
		
		float[] gradInputData = new float[]{
				16.0f, 24.0f, 24.0f, 24.0f, 16.0f, 24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 
				24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 
				16.0f, 24.0f, 24.0f, 24.0f, 16.0f, 24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 
				36.0f, 54.0f, 54.0f, 54.0f, 36.0f, 36.0f, 54.0f, 54.0f, 54.0f, 36.0f, 
				36.0f, 54.0f, 54.0f, 54.0f, 36.0f, 24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 
				16.0f, 24.0f, 24.0f, 24.0f, 16.0f, 24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 
				24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 
				16.0f, 24.0f, 24.0f, 24.0f, 16.0f, 16.0f, 24.0f, 24.0f, 24.0f, 16.0f, 
				24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 
				24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 16.0f, 24.0f, 24.0f, 24.0f, 16.0f, 
				24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 36.0f, 54.0f, 54.0f, 54.0f, 36.0f, 
				36.0f, 54.0f, 54.0f, 54.0f, 36.0f, 36.0f, 54.0f, 54.0f, 54.0f, 36.0f, 
				24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 16.0f, 24.0f, 24.0f, 24.0f, 16.0f, 
				24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 
				24.0f, 36.0f, 36.0f, 36.0f, 24.0f, 16.0f, 24.0f, 24.0f, 24.0f, 16.0f
		};
		Tensor expGradInput = new Tensor(gradInputData, 2, 3, 5, 5);
		
		float[] expDeltaData = new float[]{
				32.0f, 40.0f, 32.0f, 40.0f, 50.0f, 40.0f, 32.0f, 40.0f, 32.0f, 48.0f, 
				60.0f, 48.0f, 60.0f, 75.0f, 60.0f, 48.0f, 60.0f, 48.0f, 32.0f, 40.0f, 
				32.0f, 40.0f, 50.0f, 40.0f, 32.0f, 40.0f, 32.0f, 32.0f, 40.0f, 32.0f, 
				40.0f, 50.0f, 40.0f, 32.0f, 40.0f, 32.0f, 48.0f, 60.0f, 48.0f, 60.0f, 
				75.0f, 60.0f, 48.0f, 60.0f, 48.0f, 32.0f, 40.0f, 32.0f, 40.0f, 50.0f, 
				40.0f, 32.0f, 40.0f, 32.0f, 32.0f, 40.0f, 32.0f, 40.0f, 50.0f, 40.0f, 
				32.0f, 40.0f, 32.0f, 48.0f, 60.0f, 48.0f, 60.0f, 75.0f, 60.0f, 48.0f, 
				60.0f, 48.0f, 32.0f, 40.0f, 32.0f, 40.0f, 50.0f, 40.0f, 32.0f, 40.0f, 
				32.0f, 32.0f, 40.0f, 32.0f, 40.0f, 50.0f, 40.0f, 32.0f, 40.0f, 32.0f, 
				48.0f, 60.0f, 48.0f, 60.0f, 75.0f, 60.0f, 48.0f, 60.0f, 48.0f, 32.0f, 
				40.0f, 32.0f, 40.0f, 50.0f, 40.0f, 32.0f, 40.0f, 32.0f, 75.0f, 75.0f
		};
		Tensor expDelta = new Tensor(expDeltaData, noOutputPlanes*(noInputPlanes*kernelDepth*kernelHeight*kernelWidth+1));
		
		testModule(sc, params, input, expOutput, gradOutput, expGradInput, expDelta);
	}
}
