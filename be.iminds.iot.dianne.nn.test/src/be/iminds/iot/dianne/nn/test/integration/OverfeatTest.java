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
package be.iminds.iot.dianne.nn.test.integration;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.dataset.Sample;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.nn.test.DianneTest;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import junit.framework.Assert;


public class OverfeatTest extends DianneTest {

	public final int TEST_SAMPLE = 2;

	public void testOverfeat() throws Exception {
		NeuralNetwork nn = deployNN("overfeat_fast");
		Dataset imagenet = getDataset("ImageNet");
		
		final Sample sample = imagenet.getSample(TEST_SAMPLE);
		final Tensor result = nn.forward(sample.input);
		
		int index = TensorOps.argmax(result);
		int expected = TensorOps.argmax(sample.target);
		Assert.assertEquals(expected, index);
	}
}
