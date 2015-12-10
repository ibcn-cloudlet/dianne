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
import be.iminds.iot.dianne.tensor.TensorFactory;

/**
 * Provides base functionality for basic neural network Modules. Extend this class
 * for creating your own non-trainable module with one previous and one next Module.
 * 
 * @author tverbele
 *
 */
public abstract class AbstractModule implements Module {

	protected final static boolean TRACE = true;
	
	// the factory for this module
	protected final TensorFactory factory;
	
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
	
	// Boolean that indicates whether the next Module is still busy processing this module output
	// Can be used to either skip or block here
	protected AtomicBoolean nextBusy = new AtomicBoolean();
	
	// Thread executor to perform calculations on
	protected ExecutorService runExecutor = Executors.newSingleThreadExecutor();
	// Thread executor to notify listeners
	protected ExecutorService listenerExecutor = Executors.newSingleThreadExecutor();

	
	// Listeners
	protected Set<ForwardListener> fwdListeners = Collections.synchronizedSet(new HashSet<ForwardListener>());
	protected Set<BackwardListener> bwListeners = Collections.synchronizedSet(new HashSet<BackwardListener>());

	// Mode
	protected EnumSet<Mode> mode = EnumSet.of(Mode.BLOCKING, Mode.WAIT_FOR_FIRST);
	
	// Allows to set a common executor for multple module instances
	public void setRunExecutorService(ExecutorService executor){
		List<Runnable> todo = this.runExecutor.shutdownNow();
		this.runExecutor = executor;
		for(Runnable r : todo){
			this.runExecutor.execute(r);
		}
	}
	
	// Allows to set a common listener executor for multple module instances
	public void setListenerExecutorService(ExecutorService executor){
		List<Runnable> todo = this.listenerExecutor.shutdownNow();
		this.listenerExecutor = executor;
		for(Runnable r : todo){
			this.listenerExecutor.execute(r);
		}
	}
	
	public AbstractModule(TensorFactory factory) {
		this.id = UUID.randomUUID();
		this.factory = factory;
	}
	
	public AbstractModule(TensorFactory factory, UUID id) {
		this.id = id;
		this.factory = factory;
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

		// notify listeners
		if(fwdListeners.size()>0)
			notifyForwardListeners();
		
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
			System.out.println("BACKWARD "+this.id+" ("+this.getClass().getName()+")  FROM "+moduleId+" "+Arrays.toString(input.dims())+" "+Arrays.toString(tags));
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
		
		// notify listeners
		if(bwListeners.size()>0)
			notifyBackwardListeners();
		
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
			fwdListenersCopy.addAll(fwdListeners);
		}
		
		if(!fwdListenersCopy.isEmpty()){
			final String[] tagsCopy = (tags == null) ? null : Arrays.copyOf(tags, tags.length);

			if(exception!=null){
				listenerExecutor.execute(()->{
					fwdListenersCopy.stream().forEach(
							f -> f.onError(id, exception, tagsCopy));
				});
			} else {
				final Tensor outputCopy = output.copyInto(null);
				
				listenerExecutor.execute(()->{
					fwdListenersCopy.stream().forEach(
							f -> f.onForward(id, outputCopy, tagsCopy));
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
			bwListenersCopy.addAll(bwListeners);
		}
		
		if(!bwListenersCopy.isEmpty()){
			final String[] tagsCopy = (tags == null) ? null : Arrays.copyOf(tags, tags.length);

			if(exception!=null){
				listenerExecutor.execute(()->{
					bwListenersCopy.stream().forEach(
							b->b.onError(id, exception, tagsCopy));
				});
			} else {
				final Tensor gradInputCopy = gradInput.copyInto(null);
				
				listenerExecutor.execute(()->{
					bwListenersCopy.stream().forEach(
							b->b.onBackward(id, gradInputCopy, tagsCopy));
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
			if(ex==null){
				m.forward(id, tensor, tags);
			} else {
				m.forward(id, ex, tags);
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
			if(ex==null){
				m.backward(id, tensor, tags);
			} else {
				m.backward(id, ex, tags);
			}
			
		}
	}
	
}
