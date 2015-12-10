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
package be.iminds.iot.dianne.nn.module.fork;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

import be.iminds.iot.dianne.api.nn.module.AbstractModule;
import be.iminds.iot.dianne.api.nn.module.Module;
import be.iminds.iot.dianne.api.nn.module.ModuleException;
import be.iminds.iot.dianne.api.nn.module.Module.Mode;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorFactory;

public abstract class Fork extends AbstractModule {

	protected Map<UUID, Tensor> outputs = new HashMap<UUID, Tensor>();
	protected Map<UUID, Tensor> gradOutputs = new HashMap<UUID, Tensor>();
	
	// this will make sure that one will wait until all prev have given input before forwarding
	// during training
	protected Map<UUID, AtomicBoolean> nextLock = new HashMap<UUID, AtomicBoolean>();
	
	protected Map<Module, AtomicBoolean> nextsBusy = new HashMap<Module, AtomicBoolean>(); 
	
	protected UUID[] nextIds;
	
	public Fork(TensorFactory factory) {
		super(factory);
	}
	
	public Fork(TensorFactory factory, UUID id) {
		super(factory, id);
	}

	@Override
	protected void callNext(){
		// call all next
		for(int i=0; i< next.length;i++){
			UUID id = nextIds[i];
			Module m = next[i];
			
			if(m!=null){
				synchronized(nextsBusy){
					nextsBusy.get(m).set(true);
				}
				
				if(exception == null){
					runExecutor.execute(new ForwardForkRunnable(m, outputs.get(id), tags));
				} else {				
					runExecutor.execute(new ForwardForkRunnable(m, exception, tags));
				}
			}
		}
	}
	
	
	
	protected synchronized void forward(final UUID moduleId, final ModuleException ex, final Tensor input, final String... tags) {
		if(TRACE){
			System.out.println("FORK "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+Arrays.toString(input.dims())+" "+Arrays.toString(tags));
		}
		
		// skip or block when nexts are not ready processing previous output of this module
		synchronized(nextsBusy){
			while(nextBusy()){
				// next is busy, either block or skip
				if(mode.contains(Mode.SKIP)){
					System.out.println("Module "+id+" skipped input");
					return;
				} else {
					// default mode BLOCKING
					try {
						nextsBusy.wait();
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

		if(fwdListeners.size()>0)
			notifyForwardListeners();
		
		if(next!=null)
			callNext();
		
	}
	
	protected void backward(final UUID moduleId, final ModuleException ex, final Tensor gradOutput, final String... tags) {
		if(TRACE){
			System.out.println("BACKWARD FORK "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+Arrays.toString(input.dims())+" "+Arrays.toString(tags));
		}
		
		this.tags = tags;
		this.gradOutputs.put(moduleId, gradOutput);
		this.exception = ex;
		
		// when wait-for-all mode, wait until all gradOutput is updated
		if(mode.contains(Mode.WAIT_FOR_ALL) && next!=null && next.length>1){
			synchronized(nextLock){
				nextLock.get(moduleId).set(true);
				for(AtomicBoolean b : nextLock.values()){
					if(!b.get()){
						return;
					}
				}
				for(AtomicBoolean b : nextLock.values()){
					b.set(false);
				}
			}
		} else if(mode.contains(Mode.WAIT_FOR_FIRST)){
			if(!moduleId.equals(nextIds[0])){
				return;
			}
		}
		
		if(exception == null){
			try {
				backward();
			} catch(Exception e){
				exception = new ModuleException(this.id, this.getClass().getName(), false, e);
			}
		}

		if(bwListeners.size()>0)
			notifyBackwardListeners();
		
		// backward on separate thread
		if(prev!=null)
			callPrevious();

	}
	
	@Override
	public void setNext(final Module... next) {
		if(next==null){
			this.next = null;
			this.nextIds = null;
		} else {
			this.next = next;
			this.nextIds = new UUID[next.length];
			this.nextLock.clear();
			this.nextsBusy.clear();
			for(int i=0;i<next.length;i++){
				if(next[i]!=null){
					UUID id = next[i].getId();
					// make sure that UUIDs are in keys
					// TODO better fix for this?
					this.nextIds[i] = id;
					this.outputs.put(id, null);
					this.gradOutputs.put(id, null);
					this.nextLock.put(id, new AtomicBoolean(false));
					this.nextsBusy.put(next[i], new AtomicBoolean(false));
				}
			}
		}
	}
	
	protected final class ForwardForkRunnable implements Runnable {
		private final Module m;
		private final String[] tags;
		private final Tensor tensor;
		private final ModuleException ex;
		
		public ForwardForkRunnable(Module m, Tensor tensor, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = tensor;
			this.ex = null;
		}
		
		public ForwardForkRunnable(Module m, ModuleException ex, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = null;
			this.ex = ex;
		}
		
		public void run(){
			if(ex==null){
				m.forward(id, tensor, tags);
			} else {
				m.forward(id, ex, tags);
			}
			synchronized(nextsBusy){
				nextsBusy.get(m).set(false);
				nextsBusy.notifyAll();
			}
		}
	}
	
	protected boolean nextBusy(){
		int i = 0;
		for(AtomicBoolean a : nextsBusy.values()){
			if(a.get()){
				return true;
			}
			i++;
		}
		return false;
	}
}
