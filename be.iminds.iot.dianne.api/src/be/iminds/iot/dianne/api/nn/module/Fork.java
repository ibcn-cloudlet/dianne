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
package be.iminds.iot.dianne.api.nn.module;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Fork provides a super class for forking modules that get one input and fork to
 * multiple next modules
 * 
 * @author tverbele
 *
 */
public abstract class Fork extends AbstractModule {

	protected Map<UUID, Tensor> outputs = new HashMap<UUID, Tensor>();
	protected Map<UUID, String[]> outputTags = new HashMap<UUID, String[]>();
	protected Map<UUID, Tensor> gradOutputs = new HashMap<UUID, Tensor>();
	
	// this will make sure that one will wait until all prev have given input before forwarding
	// during training
	protected Map<UUID, AtomicBoolean> nextLock = new HashMap<UUID, AtomicBoolean>();
	
	protected Map<Module, AtomicBoolean> nextsBusy = new HashMap<Module, AtomicBoolean>(); 
	
	protected Map<UUID, Tensor> outputsListenersCopy = new HashMap<UUID, Tensor>();
	
	protected UUID[] nextIds;
	
	public Fork() {
		this(true);
	}
	
	public Fork(UUID id) {
		this(id, true);
	}
	
	public Fork(boolean waitForAll) {
		super();
		updateMode(waitForAll);
	}
	
	public Fork(UUID id, boolean waitForAll) {
		super(id);
		updateMode(waitForAll);
	}
	
	private void updateMode(boolean waitForAll) {
		mode.remove(waitForAll ? Mode.FORWARD_ON_CHANGE : Mode.WAIT_FOR_ALL);
		mode.add(waitForAll ? Mode.WAIT_FOR_ALL : Mode.FORWARD_ON_CHANGE);
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
	
	@Override
	protected void callPrevious(){
		// merge tags
		HashSet<String> mergedTags = new HashSet<>();
		for(int i=0; i< next.length;i++){
			UUID id = nextIds[i];
			String[] t = outputTags.get(id);
			if(t != null){
				for(String tag : t){
					mergedTags.add(tag);
				}
			}
		}
		this.tags = mergedTags.toArray(new String[mergedTags.size()]);
		super.callPrevious();
	}
		
	@Override
	protected synchronized void forward(final UUID moduleId, final ModuleException ex, final Tensor input, final String... tags) {
		if(TRACE){
			System.out.println("FORK "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+input+" "+Arrays.toString(tags));
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
	
	@Override
	protected void backward(final UUID moduleId, final ModuleException ex, final Tensor gradOutput, final String... tags) {
		if(TRACE){
			System.out.println("BACKWARD FORK "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+gradOutput+" "+Arrays.toString(tags));
		}
		
		this.outputTags.put(moduleId, tags);
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
		for(AtomicBoolean a : nextsBusy.values()){
			if(a.get()){
				return true;
			}
		}
		return false;
	}
	
	protected void notifyForwardListeners(){
		if(output != null)
			super.notifyForwardListeners();
		
		final List<ForwardListener> fwdListenersCopy = new ArrayList<ForwardListener>();
		synchronized(fwdListeners){
			if(forwardListenersBusy){
				if(mode.contains(Mode.SKIP)){
					return;
				} else {
					try {
						fwdListeners.wait();
					} catch (InterruptedException e) {}
				}
			}
			
			fwdListenersCopy.addAll(fwdListeners);
		}
		
		if(!fwdListenersCopy.isEmpty()){
			
			final String[] tagsCopy = (tags == null) ? null : Arrays.copyOf(tags, tags.length);

			if(exception!=null){
				listenerExecutor.execute(()->{
					fwdListenersCopy.stream().forEach(
							f -> f.onError(id, exception, tagsCopy));
					
					synchronized(fwdListeners){
						forwardListenersBusy = false;
						fwdListeners.notifyAll();
					}
				});
			} else {
				outputs.entrySet().stream().forEach(e -> outputsListenersCopy.put(e.getKey(), e.getValue().copyInto(outputsListenersCopy.get(e.getKey()))));
				for(Entry<UUID,Tensor> e : outputsListenersCopy.entrySet()){
					final int[] dims = e.getValue().dims();
					listenerExecutor.execute(()->{
						fwdListenersCopy.stream().forEach(
								f -> {
									try {
										e.getValue().reshape(dims);
										//TODO mark which output?
										f.onForward(id, e.getValue(), tagsCopy);
									} catch(Throwable t){
										System.out.println(t.getMessage());
									}
								});
						
						synchronized(fwdListeners){
							forwardListenersBusy = false;
							fwdListeners.notifyAll();
						}
					});
				}
			}
		}
	}
}
