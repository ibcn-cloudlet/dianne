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
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import be.iminds.iot.dianne.tensor.Tensor;

/**
 * Provides base functionality for basic neural network Modules. Extend this class
 * for creating your own non-trainable module with one previous and one next Module.
 * 
 * @author tverbele
 *
 */
public abstract class AbstractModule implements Module {

	// trace output for debugging
	public static boolean TRACE = false;
	private long t1,t2;
	
	// the UUID of this module
	protected final UUID id;
	
	// the tags to forward/backward
	// these are set in each forward/backward call and can be
	// adapter/filtered by the actual module implementations
	protected String[] tags;
	
	// the latest input given by the previous module
	// contains the Tensor (reference) given by previous
	protected Tensor input;
	// Tensor to put the results in of backward calculation:
	// calculates the input gradients based on the latest output gradients
	protected Tensor gradInput;

	// Tensor to put the results from from forward calculation
	// calculates the feed forward of the module
	protected Tensor output;
	// the latest gradOutputs given by previous modules
	// contains the Tensor (reference) given by previous
	protected Tensor gradOutput;
	
	protected ModuleException exception;
	
	// The next module reference
	protected Module[] next;
	// The prev module references
	protected Module[] prev;
	
	// flag to indicate whether this module is currently used in training (set to true after backward is called)
	// public visibility to be able to set this in junit test
	// TODO use a get/setMethod instead that is controlled from learner?
	public boolean train = false;
	
	// Boolean that indicates whether the next Module is still busy processing this module output
	// Can be used to either skip or block here
	protected AtomicBoolean nextBusy = new AtomicBoolean();
	
	// Thread executor to perform calculations on
	protected static ExecutorService runExecutor = Executors.newCachedThreadPool();
	// Thread executor to notify listeners
	protected static ExecutorService listenerExecutor = Executors.newSingleThreadExecutor();

	
	// Listeners
	protected Set<ForwardListener> fwdListeners = Collections.synchronizedSet(new HashSet<ForwardListener>());
	protected Set<BackwardListener> bwListeners = Collections.synchronizedSet(new HashSet<BackwardListener>());
	
	// Avoid overload of Tensor creations for the listeners, keep one copy only
	private Tensor outputListenersCopy;
	private Tensor gradInputListenersCopy;
	protected volatile boolean forwardListenersBusy = false;
	protected volatile boolean backwardListenersBusy = false;
	
	// Mode
	protected EnumSet<Mode> mode = EnumSet.of(Mode.BLOCKING, Mode.WAIT_FOR_ALL);
	
	public AbstractModule() {
		this.id = UUID.randomUUID();
	}
	
	public AbstractModule(UUID id) {
		this.id = id;
	}
	
	@Override
	public UUID getId() {
		return id;
	}
	
	protected void callNext(){
		synchronized(nextBusy){
			nextBusy.set(true);
		}
		// default AbstractModule just assumes one next and one previous, use Fork otherwise
		if(exception==null)
			runExecutor.execute(new ForwardRunnable(next[0], output, tags));
		else 
			runExecutor.execute(new ForwardRunnable(next[0], exception, tags));
	}
	
	protected void callPrevious(){
		// default AbstractModule just assumes one next and one previous, use Join otherwise
		if(exception==null){
			runExecutor.execute(new BackwardRunnable(prev[0], gradInput, tags));
		} else {
			runExecutor.execute(new BackwardRunnable(prev[0], exception, tags));
		}
	}
	
	public void forward(final UUID moduleId, final Tensor input, final String... tags){
		forward(moduleId, null, input, tags);
	}

	public void forward(final UUID moduleId, final ModuleException ex, final String... tags){
		forward(moduleId, ex, null, tags);
	}
	
	protected synchronized void forward(final UUID moduleId, final ModuleException ex, final Tensor input, final String... tags) {
		if(TRACE){
			System.out.println("FORWARD "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+input+" "+Arrays.toString(tags));
			t1 = System.nanoTime();
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

		// notify listeners
		if(fwdListeners.size()>0)
			notifyForwardListeners();
		
		if(TRACE){
			t2 = System.nanoTime();
			System.out.println("FORWARD "+this.id+" ("+this.getClass().getName()+") DONE in "+(t2-t1)+" ns");
		}

		// set train to false if only forward is called!
		train = false;
		
		// dispatch to next
		if(next!=null)
			callNext();

	}
	
	protected abstract void forward();

	public void backward(final UUID moduleId, final Tensor gradOutput, final String... tags) {
		backward(moduleId, null, gradOutput, tags); 
	}

	public void backward(final UUID moduleId, final ModuleException ex, final String... tags) {
		backward(moduleId, ex, null, tags);
	}
	
	protected synchronized void backward(final UUID moduleId, final ModuleException ex, final Tensor gradOutput, final String... tags) {
		if(TRACE){
			System.out.println("BACKWARD "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+gradOutput+" "+Arrays.toString(tags));
			t1 = System.nanoTime();
		}
		
		// backward is called, mark train true
		this.train = true;
		
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
		
		// notify listeners
		if(bwListeners.size()>0)
			notifyBackwardListeners();
		
		if(TRACE){
			t2 = System.nanoTime();
			System.out.println("BACKWARD "+this.id+" ("+this.getClass().getName()+") DONE in "+(t2-t1)+" ns");
		}
		
		// dispatch to previous
		if(prev!=null)
			callPrevious();
	}
	
	protected abstract void backward();

	@Override
	public void setNext(final Module... next) {
		this.next = next;
	}

	@Override
	public void setPrevious(final Module... prev) {
		this.prev = prev;
	}

	@Override
	public void setMode(EnumSet<Mode> mode){
		this.mode = mode;
	}
	
	public void addForwardListener(ForwardListener listener){
		fwdListeners.add(listener);
	}
	
	public void removeForwardListener(ForwardListener listener){
		fwdListeners.remove(listener);
	}
	
	protected void notifyForwardListeners(){
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
				outputListenersCopy = output.copyInto(outputListenersCopy);
				final int[] dims = output.dims();
				
				listenerExecutor.execute(()->{
					fwdListenersCopy.stream().forEach(
							f -> {
								try {
									outputListenersCopy.reshape(dims);
									f.onForward(id, outputListenersCopy, tagsCopy);
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
	
	public void addBackwardListener(BackwardListener listener){
		bwListeners.add(listener);
	}
	
	public void removeBackwardListener(BackwardListener listener){
		bwListeners.remove(listener);
	}
	
	protected void notifyBackwardListeners(){
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
				gradInputListenersCopy = gradInput==null ? null : gradInput.copyInto(gradInputListenersCopy);
				final int[] dims = gradInput==null ? null : gradInput.dims();
				
				listenerExecutor.execute(()->{
					bwListenersCopy.stream().forEach(
							b-> {
								try {
									if(dims!=null)
										gradInputListenersCopy.reshape(dims);
									
									b.onBackward(id, gradInputListenersCopy, tagsCopy);
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
	
	protected final class ForwardRunnable implements Runnable {
		private final Module m;
		private final String[] tags;
		private final Tensor tensor;
		private final ModuleException ex;
		
		public ForwardRunnable(Module m, Tensor tensor, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = tensor;
			this.ex = null;
		}
		
		public ForwardRunnable(Module m, ModuleException ex, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = null;
			this.ex = ex;
		}
		
		public void run(){
			try {
				if(ex==null){
					m.forward(id, tensor, tags);
				} else {
					m.forward(id, ex, tags);
				}
			} catch(Exception e){
				// this means there is a problem forwarding to the next module ...
				// just ignore? no way to feed error back
			}

			synchronized(nextBusy){
				nextBusy.set(false);
				nextBusy.notifyAll();
			}
		}
	}
	
	protected final class BackwardRunnable implements Runnable {
		private final Module m;
		private final String[] tags;
		private final Tensor tensor;
		private final ModuleException ex;
		
		public BackwardRunnable(Module m, Tensor tensor, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = tensor;
			this.ex = null;
		}
		
		public BackwardRunnable(Module m, ModuleException ex, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = null;
			this.ex = ex;
		}
		
		public void run(){
			try {
				if(ex==null){
					m.backward(id, tensor, tags);
				} else {
					m.backward(id, ex, tags);
				}
			} catch(Exception e){
				// this means there is a problem backwarding to the previous module ...
				// just ignore? no way to feed error back
			}
			
		}
	}
	
	public void setProperty(String key, Object val){
		throw new UnsupportedOperationException("Key "+key+" cannot be set for module "+id);
	}

}
