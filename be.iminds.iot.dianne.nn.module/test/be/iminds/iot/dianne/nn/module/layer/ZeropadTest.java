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

public class ZeropadTest extends ModuleTest{

	@Test
	public void testZeropad1() throws InterruptedException {
		Zeropad z = new Zeropad(1, 1);
		
		
		Tensor input = new Tensor(2,2);
		input.fill(1.0f);
		
		Tensor gradOutput = new Tensor(4,4);
		gradOutput.fill(1.0f);

		Tensor expOutput = new Tensor(4,4);
		expOutput.fill(0.0f);
		expOutput.set(1.0f, 1,1);
		expOutput.set(1.0f, 1,2);
		expOutput.set(1.0f, 2,1);
		expOutput.set(1.0f, 2,2);
		
		Tensor expGradInput = new Tensor(2, 2);
		expGradInput.fill(1.0f);
		
		testModule(z, input, expOutput, gradOutput, expGradInput);
	}
	
	@Test
	public void testZeropad2() throws InterruptedException {
		Zeropad z = new Zeropad(1, 1);
		
		
		Tensor input = new Tensor(2, 2, 2);
		input.fill(1.0f);
		
		Tensor gradOutput = new Tensor(2, 4, 4);
		gradOutput.fill(1.0f);

		Tensor expOutput = new Tensor(2, 4, 4);
		expOutput.fill(0.0f);
		expOutput.set(1.0f, 0, 1, 1);
		expOutput.set(1.0f, 0, 1, 2);
		expOutput.set(1.0f, 0, 2, 1);
		expOutput.set(1.0f, 0, 2, 2);
		expOutput.set(1.0f, 1, 1, 1);
		expOutput.set(1.0f, 1, 1, 2);
		expOutput.set(1.0f, 1, 2, 1);
		expOutput.set(1.0f, 1, 2, 2);
		
		
		Tensor expGradInput = new Tensor(2, 2, 2);
		expGradInput.fill(1.0f);
		
		testModule(z, input, expOutput, gradOutput, expGradInput);
	}

}
