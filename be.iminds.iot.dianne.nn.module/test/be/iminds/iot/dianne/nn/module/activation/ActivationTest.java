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
package be.iminds.iot.dianne.nn.module.activation;

import org.junit.Before;
import org.junit.Test;

import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.nn.module.ModuleTest;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;

public class ActivationTest extends ModuleTest {

	private Tensor input;
	private Tensor gradOutput;

	private Tensor inputBatch;
	private Tensor gradOutputBatch;
	
	@Before
	public void init() {
		input = new Tensor(11);
		int v = -5;
		for (int i = 0; i < 11; i++) {
			input.set(v, i);
			v++;
		}

		inputBatch = new Tensor(2, 11);
		input.copyInto(inputBatch.select(0, 0));
		input.copyInto(inputBatch.select(0, 1));
		
		gradOutput = new Tensor(11);
		gradOutput.fill(1.0f);
		
		gradOutputBatch = new Tensor(2,11);
		gradOutputBatch.fill(1.0f);
	}

	@Test
	public void testSigmoid() throws Exception {
		float[] eo = new float[] { 0.0067f, 0.0180f, 0.0474f, 0.1192f, 0.2689f,
				0.5000f, 0.7311f, 0.8808f, 0.9526f, 0.9820f, 0.9933f };
		Tensor expOutput = new Tensor(eo, 11);

		float[] eg = new float[] { 0.0066f, 0.0177f, 0.0452f, 0.1050f, 0.1966f,
				0.2500f, 0.1966f, 0.1050f, 0.0452f, 0.0177f, 0.0066f };
		Tensor expGradInput = new Tensor(eg, 11);

		Module m = new Sigmoid();
		testModule(m, input, expOutput, gradOutput, expGradInput);
	}
	
	@Test
	public void testSigmoidBatch() throws Exception {
		float[] eo = new float[] { 0.0067f, 0.0180f, 0.0474f, 0.1192f, 0.2689f,
				0.5000f, 0.7311f, 0.8808f, 0.9526f, 0.9820f, 0.9933f,
				
				0.0067f, 0.0180f, 0.0474f, 0.1192f, 0.2689f,
				0.5000f, 0.7311f, 0.8808f, 0.9526f, 0.9820f, 0.9933f};
		Tensor expOutput = new Tensor(eo, 2, 11);

		float[] eg = new float[] { 0.0066f, 0.0177f, 0.0452f, 0.1050f, 0.1966f,
				0.2500f, 0.1966f, 0.1050f, 0.0452f, 0.0177f, 0.0066f ,
				
				0.0066f, 0.0177f, 0.0452f, 0.1050f, 0.1966f,
				0.2500f, 0.1966f, 0.1050f, 0.0452f, 0.0177f, 0.0066f};
		Tensor expGradInput = new Tensor(eg, 2, 11);

		Module m = new Sigmoid();
		testModule(m, inputBatch, expOutput, gradOutputBatch, expGradInput);
	}

	@Test
	public void testTanh() throws Exception {
		float[] eo = new float[] { -0.9999f, -0.9993f, -0.9951f, -0.9640f,
				-0.7616f, 0.0000f, 0.7616f, 0.9640f, 0.9951f, 0.9993f, 0.9999f };
		Tensor expOutput = new Tensor(eo, 11);

		float[] eg = new float[] { 0.0002f, 0.0013f, 0.0099f, 0.0707f, 0.4200f,
				1.0000f, 0.4200f, 0.0707f, 0.0099f, 0.0013f, 0.0002f };
		Tensor expGradInput = new Tensor(eg, 11);

		Module m = new Tanh();
		testModule(m, input, expOutput, gradOutput, expGradInput);
	}
	
	@Test
	public void testTanhBatch() throws Exception {
		float[] eo = new float[] { -0.9999f, -0.9993f, -0.9951f, -0.9640f,
				-0.7616f, 0.0000f, 0.7616f, 0.9640f, 0.9951f, 0.9993f, 0.9999f,
				
				-0.9999f, -0.9993f, -0.9951f, -0.9640f,
				-0.7616f, 0.0000f, 0.7616f, 0.9640f, 0.9951f, 0.9993f, 0.9999f};
		Tensor expOutput = new Tensor(eo, 2, 11);

		float[] eg = new float[] { 0.0002f, 0.0013f, 0.0099f, 0.0707f, 0.4200f,
				1.0000f, 0.4200f, 0.0707f, 0.0099f, 0.0013f, 0.0002f,
				
				0.0002f, 0.0013f, 0.0099f, 0.0707f, 0.4200f,
				1.0000f, 0.4200f, 0.0707f, 0.0099f, 0.0013f, 0.0002f};
		Tensor expGradInput = new Tensor(eg, 2, 11);

		Module m = new Tanh();
		testModule(m, inputBatch, expOutput, gradOutputBatch, expGradInput);
	}

	@Test
	public void testReLU() throws Exception {
		float[] eo = new float[] { 0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f };
		Tensor expOutput = new Tensor(eo, 11);

		float[] eg = new float[] { 0f, 0f, 0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f, 1f };

		Tensor expGradInput = new Tensor(eg, 11);

		Module m = new ReLU();
		testModule(m, input, expOutput, gradOutput, expGradInput);
	}
	
	@Test
	public void testReLUBatch() throws Exception {
		float[] eo = new float[] { 0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f,
				0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f};
		Tensor expOutput = new Tensor(eo, 2, 11);

		float[] eg = new float[] { 0f, 0f, 0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f, 1f,
				0f, 0f, 0f, 0f, 0f, 0f, 1f, 1f, 1f, 1f, 1f};

		Tensor expGradInput = new Tensor(eg, 2, 11);

		Module m = new ReLU();
		testModule(m, inputBatch, expOutput, gradOutputBatch, expGradInput);
	}
	
	@Test
	public void testPReLU() throws Exception {
		float[] eo = new float[] { -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0f, 1f, 2f, 3f, 4f, 5f };
		Tensor expOutput = new Tensor(eo, 11);

		float[] eg = new float[] { 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 1f, 1f, 1f, 1f, 1f };
		Tensor expGradInput = new Tensor(eg, 11);

		// TODO also test accGrad!!!
		
		Module m = new PReLU(0.1f);
		testModule(m, input, expOutput, gradOutput, expGradInput);
	}

	@Test
	public void testSoftmax() throws Exception {
		float[] eo = new float[] { 0.0000f, 0.0001f, 0.0002f, 0.0006f, 0.0016f,
				0.0043f, 0.0116f, 0.0315f, 0.0855f, 0.2325f, 0.6321f };
		Tensor expOutput = new Tensor(eo, 11);

		float[] eg = new float[] { 0.0003e-17f, 0.0009e-17f, 0.0024e-17f,
				0.0065e-17f, 0.0175e-17f, 0.0475e-17f, 0.1287e-17f,
				0.3495e-17f, 0.9498e-17f, 2.5816e-17f, 7.0175e-17f };

		Tensor expGradInput = new Tensor(eg, 11);

		Module m = new Softmax();
		testModule(m, input, expOutput, gradOutput, expGradInput);
	}
	
	@Test
	public void testSoftmaxBatch() throws Exception {
		float[] eo = new float[] { 0.0000f, 0.0001f, 0.0002f, 0.0006f, 0.0016f,
				0.0043f, 0.0116f, 0.0315f, 0.0855f, 0.2325f, 0.6321f,
				
				0.0000f, 0.0001f, 0.0002f, 0.0006f, 0.0016f,
				0.0043f, 0.0116f, 0.0315f, 0.0855f, 0.2325f, 0.6321f};
		Tensor expOutput = new Tensor(eo, 2, 11);

		float[] eg = new float[] { 0.0003e-17f, 0.0009e-17f, 0.0024e-17f,
				0.0065e-17f, 0.0175e-17f, 0.0475e-17f, 0.1287e-17f,
				0.3495e-17f, 0.9498e-17f, 2.5816e-17f, 7.0175e-17f,
				
				0.0003e-17f, 0.0009e-17f, 0.0024e-17f,
				0.0065e-17f, 0.0175e-17f, 0.0475e-17f, 0.1287e-17f,
				0.3495e-17f, 0.9498e-17f, 2.5816e-17f, 7.0175e-17f};

		Tensor expGradInput = new Tensor(eg, 2, 11);

		Module m = new Softmax();
		testModule(m, inputBatch, expOutput, gradOutputBatch, expGradInput);
	}
	
	@Test
	public void testLogSoftmax() throws Exception {
		float[] eo = new float[] { 
				-10.458658f, -9.458658f, -8.458658f, -7.458658f, -6.458658f,
				-5.458658f, -4.458658f, -3.4586585f, -2.4586585f, -1.4586585f, -0.45865846f };
		Tensor expOutput = new Tensor(eo, 11);

		float[] eg = new float[] {0.99968433f, 0.9991419f, 0.9976674f,
				0.99365926f, 0.9827641f, 0.95314807f, 0.87264323f,
				0.6538085f, 0.058953933f, -1.5580285f, -5.953442f};

		Tensor expGradInput = new Tensor(eg, 11);

		Module m = new LogSoftmax();
		testModule(m, input, expOutput, gradOutput, expGradInput);
	}
}
