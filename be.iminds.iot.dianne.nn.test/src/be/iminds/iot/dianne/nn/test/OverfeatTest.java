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
package be.iminds.iot.dianne.nn.test;

import org.osgi.framework.ServiceReference;

import be.iminds.iot.dianne.api.dataset.Dataset;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import junit.framework.Assert;


public class OverfeatTest extends AbstractDianneTest {

	public final int TEST_SAMPLE = 2;

	private Dataset imagenet;
	
	public void setUp() throws Exception {
    	super.setUp();
    	
    	ServiceReference[] rds = context.getAllServiceReferences(Dataset.class.getName(), null);
    	for(ServiceReference rd : rds){
    		Dataset d = (Dataset) context.getService(rd);
    		if(d.getName().equals("ImageNet")){
    			imagenet = d;
    		}
    	}
    }
	
	public void testOverfeat() throws Exception {
		NeuralNetwork nn = deployNN("overfeat_fast");
		
		final Tensor sample = imagenet.getInputSample(TEST_SAMPLE);		
		final Tensor result = nn.forward(sample);
		
		int index = TensorOps.argmax(result);
		int expected = TensorOps.argmax(imagenet.getOutputSample(TEST_SAMPLE));
		Assert.assertEquals(expected, index);
	}
}
