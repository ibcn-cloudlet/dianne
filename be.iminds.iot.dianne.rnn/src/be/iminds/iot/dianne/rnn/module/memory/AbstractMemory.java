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
package be.iminds.iot.dianne.rnn.module.memory;

import java.util.Arrays;
import java.util.UUID;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Memory;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class AbstractMemory extends AbstractModule implements Memory {

	protected final Tensor memory;
	
	public AbstractMemory(TensorFactory factory, int size) {
		super(factory);
		this.memory = factory.createTensor(size);
		this.memory.fill(0.0f);
	}

	public AbstractMemory(TensorFactory factory, UUID id, int size) {
		super(factory, id);
		this.memory = factory.createTensor(size);
		this.memory.fill(0.0f);
	}
	
	public AbstractMemory(TensorFactory factory, Tensor memory) {
		super(factory);
		this.memory = memory;
	}

	public AbstractMemory(TensorFactory factory, UUID id, Tensor memory) {
		super(factory, id);
		this.memory = memory;
	}

	protected synchronized void forward(final UUID moduleId, final ModuleException ex, final Tensor input, final String... tags) {
		if(TRACE){
			System.out.println("FORWARD "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+Arrays.toString(input.dims())+" "+Arrays.toString(tags));
		}
		// skip or block when next is not ready processing previous output of this module
		synchronized(nextBusy){
			if(nextBusy.get()){
				// next is busy, either block or skip
				if(mode.contains(Mode.SKIP)){
					System.out.println("Module "+id+" skipped input");
					return;
				} else {
					// default mode BLOCKING
					try {
						nextBusy.wait();
					} catch (InterruptedException e) {
					}
				}
			}
		}
		
		this.input = input;
		this.tags = tags;
		this.exception = ex;
		
		// calculates new outputs
		if(exception == null){
			try {
				if(output!=null){
					synchronized(output){
						synchronized(input){
							forward();
						}
					}
				} else {
					synchronized(input){
						forward();
					}
				}
			} catch(Exception e){
				exception = new ModuleException(this.id, this.getClass().getName(), true, e);
			}
		} 
	}
	
	@Override
	protected void forward() {
		updateMemory();
	}
	
	// 2 step forward here : first update memory from input, on trigger update output from memory
	protected abstract void updateMemory();
	protected abstract void updateOutput();
	
	@Override
	public synchronized void triggerForward() {
		updateOutput();
		
		// notify listeners
		if(fwdListeners.size()>0)
			notifyForwardListeners();
		
		// dispatch to next
		if(next!=null)
			callNext();
	}

	
	protected synchronized void backward(final UUID moduleId, final ModuleException ex, final Tensor gradOutput, final String... tags) {
		if(TRACE){
			System.out.println("BACKWARD "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+Arrays.toString(gradOutput.dims())+" "+Arrays.toString(tags));
		}
		
		this.gradOutput = gradOutput;
		this.tags = tags;
		this.exception = ex;
		
		// calculates new gradInputs
		if(ex==null){
			try {
				backward();
			} catch(Exception e){
				exception = new ModuleException(this.id, this.getClass().getName(), false, e);
			}
		}
	}

	@Override
	public synchronized void triggerBackward() {
		// notify listeners
		if(bwListeners.size()>0)
			notifyBackwardListeners();
		
		// dispatch to previous
		if(prev!=null)
			callPrevious();
	}

	@Override
	public Tensor getMemory() {
		return memory;
	}

	@Override
	public void setMemory(Tensor memory) {
		memory.copyInto(this.memory);
	}
	
}
