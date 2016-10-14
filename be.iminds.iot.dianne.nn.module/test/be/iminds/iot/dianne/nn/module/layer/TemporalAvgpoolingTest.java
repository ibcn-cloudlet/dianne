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

public class TemporalAvgpoolingTest extends ModuleTest{

	@Test
	public void testTemporalAvgpooling1() throws InterruptedException {
		AvgPooling ap = new AvgPooling(2, 2);
		
		// temporal has "feature" dim first and "temporal" dim second
		Tensor input = new Tensor(2,6);
		input.fill(1.0f);
		input.set(2.0f, 0, 1);
		input.set(3.0f, 1, 4);
		
		Tensor gradOutput = new Tensor(2, 3);
		gradOutput.fill(1.0f);
		
		Tensor expOutput = new Tensor(2, 3);
		expOutput.fill(1.0f);
		expOutput.set(1.5f, 0, 0);
		expOutput.set(2.0f, 1, 2);
		
		Tensor expGradInput = new Tensor(2, 6);
		expGradInput.fill(0.5f);

		
		testModule(ap, input, expOutput, gradOutput, expGradInput);
	}
	


}
