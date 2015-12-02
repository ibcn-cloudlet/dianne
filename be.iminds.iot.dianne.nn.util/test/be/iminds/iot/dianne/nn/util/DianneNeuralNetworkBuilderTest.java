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
 *     Tim Verbelen, Steven Bohez
 *******************************************************************************/
package be.iminds.iot.dianne.nn.util;

import org.junit.Test;

import be.iminds.iot.dianne.api.nn.module.dto.NeuralNetworkDTO;
import be.iminds.iot.dianne.nn.util.DianneNeuralNetworkBuilder.Activation;

public class DianneNeuralNetworkBuilderTest {

	
	@Test
	public void testBuilder(){
		DianneNeuralNetworkBuilder builder = new DianneNeuralNetworkBuilder("Test");
		builder.addLinear(784, 20);
		builder.addReLU();
		builder.addLinear(20, 10);
		builder.addSoftmax();
		NeuralNetworkDTO nn = builder.create();
		System.out.println(DianneJSONConverter.toJsonString(nn, true));
	}
	
	@Test
	public void testMLP(){
		NeuralNetworkDTO nn = DianneNeuralNetworkBuilder.createMLP("MLP", 784, 10, Activation.Sigmoid, 100, 20);
		System.out.println(DianneJSONConverter.toJsonString(nn, true));
	}
}
