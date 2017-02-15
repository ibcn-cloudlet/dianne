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

public class InvertTest extends ModuleTest{

	@Test
	public void testInvert() throws InterruptedException {
		Invert i = new Invert();
		
		Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, 4);
		
		Tensor gradOutput = new Tensor(4);
		gradOutput.fill(1.0f);

		Tensor expOutput = new Tensor(new float[]{0.9f, 0.8f, 0.7f, 0.6f}, 4);
		
		Tensor expGradInput = new Tensor(4);
		expGradInput.fill(-1f);
		
		testModule(i, input, expOutput, gradOutput, expGradInput);
	}

}
