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

public class TemporalMaxpoolingTest extends ModuleTest{

	@Test
	public void testTemporalMaxpooling1() throws InterruptedException {
		MaxPooling mp = new MaxPooling(2, 2);
		
		// temporal has "temporal" dim first and "feature" dim second
		Tensor input = new Tensor(6,2);
		input.fill(1.0f);
		input.set(2.0f, 1, 0);
		input.set(3.0f, 4, 1);
		
		Tensor gradOutput = new Tensor(3, 2);
		gradOutput.fill(1.0f);
		
		Tensor expOutput = new Tensor(3, 2);
		expOutput.fill(1.0f);
		expOutput.set(2.0f, 0, 0);
		expOutput.set(3.0f, 2, 1);
		
		float[] gradInData = new float[]{0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f};
		Tensor expGradInput = new Tensor(gradInData, 6, 2);
		
		testModule(mp, input, expOutput, gradOutput, expGradInput);
	}
	


}
