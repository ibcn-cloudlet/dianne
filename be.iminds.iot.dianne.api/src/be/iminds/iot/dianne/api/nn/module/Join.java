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
 * Join provides a super class for joining modules that get multiple inputs and join to
 * a single next module
 * 
 * @author tverbele
 *
 */
public abstract class Join extends AbstractModule {

	protected Map<UUID, Tensor> inputs = new HashMap<UUID, Tensor>();
	protected Map<UUID, String[]> inputTags = new HashMap<UUID, String[]>();
	protected Map<UUID, Tensor> gradInputs = new HashMap<UUID, Tensor>();
	
	// this will make sure that one will wait until all prev have given input before forwarding 
	// during training
	protected Map<UUID, AtomicBoolean> prevLock = new HashMap<UUID, AtomicBoolean>();
	
	// might need the order of id
	protected UUID[] prevIds;
	
	protected Map<UUID, Tensor> gradInputsListenersCopy = new HashMap<UUID, Tensor>();

	
	public Join() {
		this(true);
	}
	
	public Join(UUID id) {
		this(id, true);
	}
	
	public Join(boolean waitForAll) {
		super();
		updateMode(waitForAll);
	}
	
	public Join(UUID id, boolean waitForAll) {
		super(id);
		updateMode(waitForAll);
	}
	
	private void updateMode(boolean waitForAll) {
		mode.remove(waitForAll ? Mode.FORWARD_ON_CHANGE : Mode.WAIT_FOR_ALL);
		mode.add(waitForAll ? Mode.WAIT_FOR_ALL : Mode.FORWARD_ON_CHANGE);
	}
	
	@Override
	protected void callNext(){
		// merge tags
		HashSet<String> mergedTags = new HashSet<>();
		for(int i=0; i< prev.length;i++){
			UUID id = prevIds[i];
			String[] t = inputTags.get(id);
			if(t != null){
				for(String tag : t){
					mergedTags.add(tag);
				}
			}
		}
		this.tags = mergedTags.toArray(new String[mergedTags.size()]);
		super.callNext();
	}
		
	@Override
	protected void callPrevious(){
		// call all previous
		for(int i=0; i< prev.length;i++){
			UUID id = prevIds[i];
			Module m = prev[i];
			
			if(m!=null){
				if(exception == null){
					runExecutor.execute(new BackwardRunnable(m, gradInputs.get(id), tags));
				} else {				
					runExecutor.execute(new BackwardRunnable(m, exception, tags));
				}
			}
		}
	}
	
	@Override
	protected synchronized void forward(final UUID moduleId, final ModuleException ex, final Tensor input, final String... tags) {
		if(TRACE){
			System.out.println("JOIN "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+input+" "+Arrays.toString(tags));
		}
		
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
		
		this.inputs.put(moduleId, input);
		this.inputTags.put(moduleId, tags);
		this.exception = ex;
		
		if(this.exception==null){
		
			// when in wait-for-all mode, wait for input from each prev
			if(mode.contains(Mode.WAIT_FOR_ALL) && prev!=null && prev.length>1){
				synchronized(prevLock){
					prevLock.get(moduleId).set(true);
					for(AtomicBoolean b : prevLock.values()){
						if(!b.get()){
							return;
						}
					}
					for(AtomicBoolean b : prevLock.values()){
						b.set(false);
					}
				}
			} else if(mode.contains(Mode.WAIT_FOR_FIRST)){
				if(!moduleId.equals(prevIds[0])){
					return;
				}
			}
			
			
			try {
				forward();
			} catch(Exception e){
				this.exception = new ModuleException(this.id, this.getClass().getName(), true, e);
			}
		
		} 

		if(fwdListeners.size()>0)
			notifyForwardListeners();
		
		if(next!=null)
			callNext();
	
	}
	
	@Override
	public void setPrevious(final Module... prev) {
		if(prev==null){
			this.prev = null;
			this.prevIds = null;
		} else {
			this.prev = prev;
			this.prevIds = new UUID[prev.length];
			for(int i=0;i<prev.length;i++){
				// make sure that UUIDs are in keys
				// TODO better fix for this?
				if(prev[i]!=null){
					UUID id = prev[i].getId();
					prevIds[i] = id;
					inputs.put(id, null);
					gradInputs.put(id, null);
					prevLock.put(id, new AtomicBoolean(false));
				}
			}
		}
	}
	
	protected void notifyBackwardListeners(){
		if(gradInput!=null)
			super.notifyBackwardListeners();
		
		final List<BackwardListener> bwListenersCopy = new ArrayList<BackwardListener>();
		synchronized(bwListeners){
			if(backwardListenersBusy){
				if(mode.contains(Mode.SKIP)){
					return;
				} else {
					try {
						bwListeners.wait();
					} catch (InterruptedException e) {}
				}
			}
			bwListenersCopy.addAll(bwListeners);
		}
		
		if(!bwListenersCopy.isEmpty()){
			final String[] tagsCopy = (tags == null) ? null : Arrays.copyOf(tags, tags.length);

			if(exception!=null){
				listenerExecutor.execute(()->{
					bwListenersCopy.stream().forEach(
							b->b.onError(id, exception, tagsCopy));
					
					synchronized(bwListeners){
						backwardListenersBusy = false;
						bwListeners.notifyAll();
					}
				});
			} else {
				gradInputs.entrySet().stream().forEach(e -> gradInputsListenersCopy.put(e.getKey(), e.getValue().copyInto(gradInputsListenersCopy.get(e.getKey()))));
				
				for(Entry<UUID,Tensor> e : gradInputsListenersCopy.entrySet()){
					final int[] dims = e.getValue().dims();
					listenerExecutor.execute(()->{
						bwListenersCopy.stream().forEach(
								b-> {
									try {
										e.getValue().reshape(dims);
										//TODO mark which input?
										b.onBackward(id, e.getValue(), tagsCopy);
									} catch(Throwable t){
										System.out.println(t.getMessage());
									}
								});
						
						synchronized(bwListeners){
							backwardListenersBusy = false;
							bwListeners.notifyAll();
						}
					});
				}
			}
		}
	}
}
