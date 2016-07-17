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
package be.iminds.iot.dianne.nn.module;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;

import be.iminds.iot.dianne.api.nn.module.BackwardListener;
import be.iminds.iot.dianne.api.nn.module.ForwardListener;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.api.nn.module.Trainable;
import be.iminds.iot.dianne.tensor.NativeTensorLoader;
import be.iminds.iot.dianne.tensor.Tensor;

public class ModuleTest {
	
	public static final boolean TRACE = true;
	
	@BeforeClass
	public static void setup() {
		NativeTensorLoader loader = new NativeTensorLoader();
		loader.activate(null);
	}
	
	protected void testModule(Module m, Tensor input, Tensor expOutput, Tensor gradOutput, 
			Tensor expGradInput) throws InterruptedException {
		try {
			Tensor output = new Tensor();
			Tensor gradInput = new Tensor();
			final List<Exception> errors = new ArrayList<>();
			
			m.addForwardListener(new ForwardListener() {
				@Override
				public void onForward(UUID moduleId, Tensor o, String... tags) {
					if(TRACE)
						System.out.println("OUTPUT "+o);
					o.copyInto(output);
					
					if(TRACE)
						System.out.println("GRAD OUT "+gradOutput);
					
					m.backward(UUID.randomUUID(), gradOutput);
				}
				
				@Override
				public void onError(final UUID moduleId, final ModuleException e, final String...tags){
					e.printStackTrace();
					errors.add(e);
					synchronized(m){
						m.notify();
					}
				}
			});
			m.addBackwardListener(new BackwardListener() {
				@Override
				public void onBackward(UUID moduleId, Tensor gi, String... tags) {
					synchronized(m) {
						if(expGradInput!=null){
							if(TRACE)
								System.out.println("GRAD IN "+gi);
							
							gi.copyInto(gradInput);
						}
						m.notify();
					}
				}
				
				@Override
				public void onError(final UUID moduleId, final ModuleException e, final String...tags){
					e.printStackTrace();
					errors.add(e);
					synchronized(m){
						m.notify();
					}
				}
			});
			
			if(TRACE)
				System.out.println("INPUT "+input);
			
			synchronized(m) {
				m.forward(UUID.randomUUID(), input);
				m.wait(1000);
				if(!errors.isEmpty()){
					Assert.fail(errors.get(0).getMessage());
				}
			}
			
			Assert.assertTrue("Wrong output", expOutput.equals(output, 0.005f));
			if(expGradInput != null)
				Assert.assertTrue("Wrong grad input", expGradInput.equals(gradInput, 0.005f));

		} catch (UnsupportedOperationException ex) {
			Assume.assumeNoException("Method not implemented yet for current configuration.", ex);
		}
	}

	
	protected void testModule(Trainable m, Tensor params, Tensor input, Tensor expOutput, Tensor gradOutput, 
			Tensor expGradInput, Tensor expDeltaParameters) throws InterruptedException {
		try {
			Tensor output = new Tensor();
			Tensor gradInput = new Tensor();
			final List<Exception> errors = new ArrayList<>();
			
			m.zeroDeltaParameters();
			if(params !=null)
				m.setParameters(params);		
			
			m.addForwardListener(new ForwardListener() {
				@Override
				public void onForward(UUID moduleId, Tensor o, String... tags) {
					if(TRACE)
						System.out.println("OUTPUT "+o);
					o.copyInto(output);
					
					if(TRACE)
						System.out.println("GRAD OUT "+gradOutput);
					
					m.backward(UUID.randomUUID(), gradOutput);
				}
				
				@Override
				public void onError(final UUID moduleId, final ModuleException e, final String...tags){
					e.printStackTrace();
					errors.add(e);
					synchronized(m){
						m.notify();
					}
				}
			});
			m.addBackwardListener(new BackwardListener() {
				@Override
				public void onBackward(UUID moduleId, Tensor gi, String... tags) {
					if(TRACE)
						System.out.println("GRAD IN "+gi);

					synchronized(m) {
						gi.copyInto(gradInput);
						m.notify();
					}
				}
				
				@Override
				public void onError(final UUID moduleId, final ModuleException e, final String...tags){
					e.printStackTrace();
					errors.add(e);
					synchronized(m){
						m.notify();
					}
				}
			});

			if(TRACE)
				System.out.println("INPUT "+input);

			synchronized(m) {
				m.forward(UUID.randomUUID(), input);
				m.wait(1000);
				if(!errors.isEmpty()){
					Assert.fail(errors.get(0).getMessage());
				}
				m.accGradParameters();
			}
			
			if(TRACE)
				System.out.println("DELTA PARAMS "+m.getDeltaParameters());
			
			Assert.assertTrue("Wrong output", expOutput.equals(output, 0.001f));
			if(expGradInput!= null)
				Assert.assertTrue("Wrong grad input", expGradInput.equals(gradInput, 0.001f));
			if(expDeltaParameters!=null)
				Assert.assertTrue("Wrong delta parameters", expDeltaParameters.equals(m.getDeltaParameters(), 0.001f));
			
		} catch (UnsupportedOperationException ex) {
			Assume.assumeNoException("Method not implemented yet for current configuration.", ex);
		} 
	}
}
