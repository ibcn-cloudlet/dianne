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
package be.iminds.iot.dianne.nn.module.forkjoin;

import java.util.UUID;
import java.util.concurrent.CountDownLatch;

import org.junit.Assert;
import org.junit.Test;

import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.nn.module.ModuleTest;
import be.iminds.iot.dianne.nn.module.fork.Split;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.io.OutputImpl;
import be.iminds.iot.dianne.tensor.Tensor;

public class SplitTest extends ModuleTest{

	public void testSplit(Split s, Tensor input, Tensor gradOutput1, Tensor gradOutput2, Tensor expOutput1, Tensor expOutput2, Tensor expGradInput) throws InterruptedException{
		UUID uuid1 = UUID.randomUUID();
		UUID uuid2 = UUID.randomUUID();
		UUID uuid3 = UUID.randomUUID();
		
		Input i = new InputImpl(uuid1);
		Output o1 = new OutputImpl(uuid2);
		Output o2 = new OutputImpl(uuid3);
		i.setNext(s);
		s.setNext(o1,o2);
		s.setPrevious(i);
		o1.setPrevious(s);
		o2.setPrevious(s);
		
		Tensor output1 = new Tensor();
		Tensor output2 = new Tensor();
		Tensor gradInput = new Tensor();
		
		CountDownLatch latch = new CountDownLatch(1);
		
		o1.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(UUID moduleId, Tensor o, String... tags) {
				if(TRACE)
					System.out.println("OUTPUT "+o);
				o.copyInto(output1);
				
				if(TRACE)
					System.out.println("GRAD OUT "+gradOutput1);
				
				s.backward(uuid2, gradOutput1);
			}
		});
		o2.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(UUID moduleId, Tensor o, String... tags) {
				if(TRACE)
					System.out.println("OUTPUT "+o);
				o.copyInto(output2);
				
				if(TRACE)
					System.out.println("GRAD OUT "+gradOutput2);
				
				s.backward(uuid3, gradOutput2);
			}
		});
		i.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(UUID moduleId, Tensor gi, String... tags) {
				if(expGradInput!=null){
					if(TRACE)
						System.out.println("GRAD IN "+gi);
						
					gi.copyInto(gradInput);
				}
				latch.countDown();
			}
		});
		
		
		if(TRACE){
			System.out.println("INPUT"+input);
		}
		
		i.forward(uuid1, input);
		latch.await();
	
		Assert.assertTrue("Wrong output 1", expOutput1.equals(output1, 0.005f));
		Assert.assertTrue("Wrong output 2", expOutput2.equals(output2, 0.005f));
		Assert.assertTrue("Wrong grad input ", expGradInput.equals(gradInput, 0.005f));
	}
	
	@Test
	public void testSplit1() throws InterruptedException {
		
		Tensor input = new Tensor(new float[]{1,2,3,4,5,6,7,8,9,10}, 10);
		
		Tensor expOutput1 = new Tensor(new float[]{1,2,3,4,5}, 5);
		Tensor expOutput2 = new Tensor(new float[]{6,7,8,9,10}, 5);
		
		Tensor gradOutput1 = new Tensor(new float[]{1,2,3,4,5}, 5);
		Tensor gradOutput2 = new Tensor(new float[]{6,7,8,9,10}, 5);

		
		Tensor expGradInput = new Tensor(new float[]{1,2,3,4,5,6,7,8,9,10}, 10);
		

		Split s = new Split(0);
		testSplit(s, input, gradOutput1, gradOutput2, expOutput1, expOutput2, expGradInput);
	}
	
	
	@Test
	public void testSplit2() throws InterruptedException {
		
		Tensor input = new Tensor(new float[]{1,2,3,4,5,6,7,8,9,10}, 10);
		
		Tensor expOutput1 = new Tensor(new float[]{1,2,3,4}, 4);
		Tensor expOutput2 = new Tensor(new float[]{5,6,7,8,9,10}, 6);
		
		Tensor gradOutput1 = new Tensor(new float[]{1,2,3,4}, 4);
		Tensor gradOutput2 = new Tensor(new float[]{5,6,7,8,9,10}, 6);

		
		Tensor expGradInput = new Tensor(new float[]{1,2,3,4,5,6,7,8,9,10}, 10);
		

		Split s = new Split(0, new int[]{4});
		testSplit(s, input, gradOutput1, gradOutput2, expOutput1, expOutput2, expGradInput);
	}

	@Test
	public void testSplit3() throws InterruptedException {
		
		Tensor input = new Tensor(new float[]{1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},2, 10);
		
		Tensor expOutput1 = new Tensor(new float[]{1,2,3,4,5,1,2,3,4,5},2, 5);
		Tensor expOutput2 = new Tensor(new float[]{6,7,8,9,10,6,7,8,9,10},2, 5);
		
		Tensor gradOutput1 = new Tensor(new float[]{1,2,3,4,5,1,2,3,4,5},2, 5);
		Tensor gradOutput2 = new Tensor(new float[]{6,7,8,9,10,6,7,8,9,10},2, 5);

		
		Tensor expGradInput = new Tensor(new float[]{1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},2, 10);
		

		Split s = new Split(0);
		testSplit(s, input, gradOutput1, gradOutput2, expOutput1, expOutput2, expGradInput);
	}
	
	@Test
	public void testSplit4() throws InterruptedException {
		
		Tensor input = new Tensor(new float[]{1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10}, 2, 10);
		
		Tensor expOutput1 = new Tensor(new float[]{1,2,3,4,1,2,3,4}, 2, 4);
		Tensor expOutput2 = new Tensor(new float[]{5,6,7,8,9,10,5,6,7,8,9,10},2, 6);
		
		Tensor gradOutput1 = new Tensor(new float[]{1,2,3,4,1,2,3,4},2, 4);
		Tensor gradOutput2 = new Tensor(new float[]{5,6,7,8,9,10,5,6,7,8,9,10},2, 6);

		
		Tensor expGradInput = new Tensor(new float[]{1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10},2, 10);
		

		Split s = new Split(0, new int[]{4});
		testSplit(s, input, gradOutput1, gradOutput2, expOutput1, expOutput2, expGradInput);
	}
}
