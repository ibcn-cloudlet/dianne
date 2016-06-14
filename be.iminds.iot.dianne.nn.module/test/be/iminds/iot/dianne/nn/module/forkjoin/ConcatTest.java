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

import org.junit.Assert;
import org.junit.Test;

import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.nn.module.ModuleTest;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.join.Concat;
import be.iminds.iot.dianne.tensor.Tensor;

public class ConcatTest extends ModuleTest{

	public void testConcat(Tensor input1, Tensor input2, Tensor gradOutput, Tensor expOutput, Tensor expGradInput1, Tensor expGradInput2) throws InterruptedException{
		UUID uuid1 = UUID.randomUUID();
		UUID uuid2 = UUID.randomUUID();
		UUID uuid3 = UUID.randomUUID();
		
		
		Input i1 = new InputImpl(uuid1);
		Input i2 = new InputImpl(uuid2);
		Concat c = new Concat(uuid3, 2);
		i1.setNext(new Module[]{c});
		i2.setNext(new Module[]{c});
		c.setPrevious(new Module[]{i1, i2});
		
		Tensor output = new Tensor();
		Tensor gradInput1 = new Tensor();
		Tensor gradInput2 = new Tensor();
		
		
		c.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(UUID moduleId, Tensor o, String... tags) {
				if(TRACE)
					System.out.println("OUTPUT "+o);
				o.copyInto(output);
				
				if(TRACE)
					System.out.println("GRAD OUT "+gradOutput);
				
				c.backward(UUID.randomUUID(), gradOutput);
			}
		});
		i1.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(UUID moduleId, Tensor gi, String... tags) {
				synchronized(i1) {
					if(expGradInput1!=null){
						if(TRACE)
							System.out.println("GRAD IN "+gi);
						
						gi.copyInto(gradInput1);
					}
					i1.notify();
				}
			}
		});
		i2.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(UUID moduleId, Tensor gi, String... tags) {
				synchronized(i2) {
					if(expGradInput2!=null){
						if(TRACE)
							System.out.println("GRAD IN "+gi);
						
						gi.copyInto(gradInput2);
					}
					i2.notify();
				}
			}
		});
		
		if(TRACE){
			System.out.println("INPUT 1"+input1);
			System.out.println("INPUT 2"+input2);
		}
		
		synchronized(i2) {
			i1.forward(UUID.randomUUID(), input1);
			i2.forward(UUID.randomUUID(), input2);
			i2.wait(1000);
		}
	
		Assert.assertTrue("Wrong output", expOutput.equals(output, 0.005f));
		Assert.assertTrue("Wrong grad input 1", expGradInput1.equals(gradInput1, 0.005f));
		Assert.assertTrue("Wrong grad input 2", expGradInput2.equals(gradInput2, 0.005f));
	}
	
	@Test
	public void testConcat1() throws InterruptedException {
		
		float[] inputData1 = new float[2*2*3];
		for(int i=0;i<inputData1.length;i++){
			inputData1[i] = i;
		}
		Tensor input1 = new Tensor(inputData1, 3,2,2);
		
		float[] inputData2 = new float[2*2*2];
		for(int i=0;i<inputData2.length;i++){
			inputData2[i] = i;
		}
		Tensor input2 = new Tensor(inputData2, 2,2,2);
		
		
		Tensor gradOutput = new Tensor(5, 2, 2);
		gradOutput.fill(1.0f);

		float[] expOutputData = new float[]{
				0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
				0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f
		};
		Tensor expOutput = new Tensor(expOutputData, 5, 2, 2);
		
		Tensor expGradInput1 = new Tensor(3, 2, 2);
		expGradInput1.fill(1.0f);
		Tensor expGradInput2 = new Tensor(2, 2, 2);
		expGradInput2.fill(1.0f);

		testConcat(input1, input2, gradOutput, expOutput, expGradInput1, expGradInput2);
	}
	
	@Test
	public void testConcat2() throws InterruptedException {
		
		float[] inputData1 = new float[2*2*2*3];
		for(int i=0;i<inputData1.length;i++){
			inputData1[i] = i;
		}
		Tensor input1 = new Tensor(inputData1, 2,3,2,2);
		
		float[] inputData2 = new float[2*2*2*2];
		for(int i=0;i<inputData2.length;i++){
			inputData2[i] = i;
		}
		Tensor input2 = new Tensor(inputData2, 2,2,2,2);
		
		
		Tensor gradOutput = new Tensor(2, 5, 2, 2);
		gradOutput.fill(1.0f);

		float[] expOutputData = new float[]{
				// t1 of batch
				0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 
				0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
				// t2 of batch
				12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
				8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
		};
		Tensor expOutput = new Tensor(expOutputData, 2, 5, 2, 2);
		
		Tensor expGradInput1 = new Tensor(2, 3, 2, 2);
		expGradInput1.fill(1.0f);
		Tensor expGradInput2 = new Tensor(2, 2, 2, 2);
		expGradInput2.fill(1.0f);

		testConcat(input1, input2, gradOutput, expOutput, expGradInput1, expGradInput2);
	}
}
