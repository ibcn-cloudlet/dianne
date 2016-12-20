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
package be.iminds.iot.dianne.nn.module.vae;

import java.util.UUID;
import java.util.concurrent.CountDownLatch;

import org.junit.Assert;
import org.junit.Test;

import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Input;
import be.iminds.iot.dianne.api.nn.module.Mimo;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.Output;
import be.iminds.iot.dianne.nn.module.ModuleTest;
import be.iminds.iot.dianne.nn.module.io.InputImpl;
import be.iminds.iot.dianne.nn.module.io.OutputImpl;
import be.iminds.iot.dianne.tensor.Tensor;

public class MultivariateGaussianTest extends ModuleTest{

	public void testMimo(Mimo m, Tensor input1, Tensor input2, Tensor gradOutput1, Tensor gradOutput2, Tensor expOutput1, Tensor expOutput2, Tensor expGradInput1, Tensor expGradInput2) throws InterruptedException{
		UUID uuid1 = UUID.randomUUID();
		UUID uuid2 = UUID.randomUUID();
		UUID uuid3 = UUID.randomUUID();
		UUID uuid4 = UUID.randomUUID();
		
		int wait = 2;
		
		Input i1 = new InputImpl(uuid1);
		i1.setNext(new Module[]{m});
		
		Input i2 = new InputImpl(uuid2);
		if(input2 != null){
			i2.setNext(new Module[]{m});
			m.setPrevious(new Module[]{i1, i2});
		} else {
			m.setPrevious(i1);
			wait = 1;
		}
		
		Output o1 = new OutputImpl(uuid3);
		o1.setPrevious(m);

		Output o2 = new OutputImpl(uuid4);
		if(expOutput2 != null){
			o2.setPrevious(m);
			m.setNext(new Module[]{o1, o2});
		} else {
			m.setNext(o1);
		}
		
		
		Tensor output1 = new Tensor();
		Tensor output2 = new Tensor();
		Tensor gradInput1 = new Tensor();
		Tensor gradInput2 = new Tensor();
		
		CountDownLatch latch = new CountDownLatch(wait);
		
		o1.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(UUID moduleId, Tensor o, String... tags) {
				if(TRACE)
					System.out.println("OUTPUT 1"+o);
				o.copyInto(output1);
				
				if(TRACE)
					System.out.println("GRAD OUT 1"+gradOutput1);
				
				o1.backward(UUID.randomUUID(), gradOutput1);
			}
		});
		
		o2.addForwardListener(new ForwardListener() {
			@Override
			public void onForward(UUID moduleId, Tensor o, String... tags) {
				if(TRACE)
					System.out.println("OUTPUT 2"+o);
				o.copyInto(output2);
				
				if(TRACE)
					System.out.println("GRAD OUT 2"+gradOutput2);
				
				o2.backward(UUID.randomUUID(), gradOutput2);
			}
		});
		
		
		i1.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(UUID moduleId, Tensor gi, String... tags) {
				if(expGradInput1!=null){
					if(TRACE)
						System.out.println("GRAD IN "+gi);
						
					gi.copyInto(gradInput1);
				}
				latch.countDown();
			}
		});
		
		i2.addBackwardListener(new BackwardListener() {
			@Override
			public void onBackward(UUID moduleId, Tensor gi, String... tags) {
				if(expGradInput2!=null){
					if(TRACE)
						System.out.println("GRAD IN "+gi);
						
					gi.copyInto(gradInput2);
				}
				latch.countDown();
			}
		});
		
		if(TRACE){
			System.out.println("INPUT 1"+input1);
			System.out.println("INPUT 2"+input2);
		}
		
		i1.forward(UUID.randomUUID(), input1);
		if(input2 != null)
			i2.forward(UUID.randomUUID(), input2);
		
		latch.await();
	
		Assert.assertTrue("Wrong output", expOutput1.equals(output1, 0.005f));
		if(expOutput2 != null)
			Assert.assertTrue("Wrong output", expOutput2.equals(output2, 0.005f));
		Assert.assertTrue("Wrong grad input 1", expGradInput1.equals(gradInput1, 0.005f));
		
		if(expGradInput2 != null)
			Assert.assertTrue("Wrong grad input 2", expGradInput2.equals(gradInput2, 0.005f));
	}
	
	@Test
	public void testMultivariateGaussian1to1() throws InterruptedException {
		Tensor t = new Tensor(new float[]{0f, 0.1f, 1f, 1.1f}, 4);

		MultivariateGaussian g = new MultivariateGaussian(UUID.randomUUID(), 2);
		testMimo(g, t, null, t, null, t, null, t, null);
	}

	@Test
	public void testMultivariateGaussian1to1Batch() throws InterruptedException {
		Tensor t = new Tensor(new float[]{0f, 0.1f, 1f, 1.1f, 
				0.2f, 0.3f, 1.2f, 1.3f}, 2, 4);

		MultivariateGaussian g = new MultivariateGaussian(UUID.randomUUID(), 2);
		testMimo(g, t, null, t, null, t, null, t, null);
	}
	
	@Test
	public void testMultivariateGaussian2to2() throws InterruptedException {
		Tensor t = new Tensor(new float[]{0f, 0.1f, 1f, 1.1f}, 4);
		Tensor mean = new Tensor(new float[]{0f, 0.1f}, 2);
		Tensor stdev = new Tensor(new float[]{1f, 1.1f}, 2);

		MultivariateGaussian g = new MultivariateGaussian(UUID.randomUUID(), 2);
		testMimo(g, mean, stdev, mean, stdev, mean, stdev, mean, stdev);
	}

	@Test
	public void testMultivariateGaussian1to2() throws InterruptedException {
		Tensor t = new Tensor(new float[]{0f, 0.1f, 1f, 1.1f}, 4);
		Tensor mean = new Tensor(new float[]{0f, 0.1f}, 2);
		Tensor stdev = new Tensor(new float[]{1f, 1.1f}, 2);

		MultivariateGaussian g = new MultivariateGaussian(UUID.randomUUID(), 2);
		testMimo(g, t, null, mean, stdev, mean, stdev, t, null);
	}
	
	@Test
	public void testMultivariateGaussian2to1() throws InterruptedException {
		Tensor t = new Tensor(new float[]{0f, 0.1f, 1f, 1.1f}, 4);
		Tensor mean = new Tensor(new float[]{0f, 0.1f}, 2);
		Tensor stdev = new Tensor(new float[]{1f, 1.1f}, 2);

		MultivariateGaussian g = new MultivariateGaussian(UUID.randomUUID(), 2);
		testMimo(g, mean, stdev, t, null, t, null, mean, stdev);
	}
	
	@Test
	public void testMultivariateGaussianSigmoid() throws InterruptedException {
		Tensor mean = new Tensor(new float[]{0f, 1f}, 2);
		Tensor stdev = new Tensor(new float[]{2f, 3f}, 2);

		Tensor smean = new Tensor(new float[]{0.5000f, 0.7311f}, 2);
		Tensor sstdev = new Tensor(new float[]{0.8808f, 0.9526f}, 2);

		Tensor gsmean = new Tensor(new float[]{0.2500f, 0.1966f}, 2);
		Tensor gsstdev = new Tensor(new float[]{0.1050f, 0.0452f}, 2);

		Tensor gOut = new Tensor(2);
		gOut.fill(1.0f);
		
		MultivariateGaussian g1 = new MultivariateGaussian(UUID.randomUUID(), 2, "Sigmoid", null);
		MultivariateGaussian g2 = new MultivariateGaussian(UUID.randomUUID(), 2, null, "Sigmoid");
		MultivariateGaussian g3 = new MultivariateGaussian(UUID.randomUUID(), 2, "sigmoid", "SigMOiD");

		
		testMimo(g1, mean, stdev, gOut, gOut, smean, stdev, gsmean, gOut);
		testMimo(g2, mean, stdev, gOut, gOut, mean, sstdev, gOut, gsstdev);
		testMimo(g3, mean, stdev, gOut, gOut, smean, sstdev, gsmean, gsstdev);
		
	}
}