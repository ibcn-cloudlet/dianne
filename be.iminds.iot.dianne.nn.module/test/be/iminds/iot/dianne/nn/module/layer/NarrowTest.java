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

public class NarrowTest extends ModuleTest{

	@Test
	public void testNarrow1() throws InterruptedException {
		Narrow n = new Narrow(1,2);	
		
		Tensor input = new Tensor(new float[]{1,2,3,4}, 4);
		
		Tensor gradOutput = new Tensor(2);
		gradOutput.fill(1.0f);

		Tensor expOutput = new Tensor(new float[]{2,3}, 2);
		
		Tensor expGradInput = new Tensor(new float[]{0, 1, 1, 0}, 4);
		
		testModule(n, input, expOutput, gradOutput, expGradInput);
	}
	
	@Test
	public void testNarrow2() throws InterruptedException {
		Narrow n = new Narrow(1,2);	
		
		Tensor input = new Tensor(new float[]{1,2,3,4,1,2,3,4}, 2, 4);
		
		Tensor gradOutput = new Tensor(2, 2);
		gradOutput.fill(1.0f);

		Tensor expOutput = new Tensor(new float[]{2,3,2,3}, 2, 2);
		
		Tensor expGradInput = new Tensor(new float[]{0, 1, 1, 0, 0, 1, 1, 0},2, 4);
		
		testModule(n, input, expOutput, gradOutput, expGradInput);
	}

}
