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
package be.iminds.iot.dianne.nn.module.preprocessing;

import org.junit.Test;

import be.iminds.iot.dianne.nn.module.ModuleTest;
import be.iminds.iot.dianne.tensor.Tensor;

public class FrameTest extends ModuleTest {

	@Test
	public void testFrame() throws Exception {
	
		Frame frame = new Frame(2, 2);
		
		float[] inputData = new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
		Tensor input = new Tensor(inputData, 3, 5);
		
		float[] expOutputData = new float[]{2.0f, 4.0f, 12.0f, 14.0f};
		Tensor expOutput = new Tensor(expOutputData, 2, 2);
		
		float[] gradOutputData = new float[]{1.0f, 1.0f, 1.0f, 1.0f};
		Tensor gradOutput = new Tensor(gradOutputData, 2, 2);

		// TODO also implement a backward for frame?
		Tensor expGradInput = null;
		
		testModule(frame, input, expOutput, gradOutput, expGradInput);
	}
	
	@Test
	public void testFrameBatch() throws Exception {
	
		Frame frame = new Frame(2, 2);
		
		float[] inputData = new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
				1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
		Tensor input = new Tensor(inputData, 2, 3, 5);
		
		float[] expOutputData = new float[]{2.0f, 4.0f, 12.0f, 14.0f, 2.0f, 4.0f, 12.0f, 14.0f};
		Tensor expOutput = new Tensor(expOutputData, 2, 2, 2);
		
		float[] gradOutputData = new float[]{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
		Tensor gradOutput = new Tensor(gradOutputData, 2, 2, 2);

		// TODO also implement a backward for frame?
		Tensor expGradInput = null;
		
		testModule(frame, input, expOutput, gradOutput, expGradInput);
	}
}
