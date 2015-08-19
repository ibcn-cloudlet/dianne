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
	protected EnumSet<Mode> mode = EnumSet.of(Mode.BLOCKING);
	
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
		runExecutor.execute(new ForwardRunnable(next[0], output, tags));
	}
	
	protected void callPrevious(){
		// default AbstractModule just assumes one next and one previous, use Join otherwise
		runExecutor.execute(new BackwardRunnable(prev[0], gradInput, tags));
	}
	
	@Override
	public void forward(final UUID moduleId, final Tensor input, final String... tags) {
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
		
		// calculates new outputs
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

		// notify listeners
		if(fwdListeners.size()>0)
			notifyForwardListeners();
		
		// dispatch to next
		if(next!=null)
			callNext();
	
	}
	
	protected abstract void forward();
	
	@Override
	public void backward(final UUID moduleId, final Tensor gradOutput, final String... tags) {
		this.gradOutput = gradOutput;
		this.tags = tags;
		
		// calculates new gradInputs
		backward();
		
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
		final Tensor outputCopy = output.copyInto(null);
		final String[] tagsCopy = (tags == null) ? null : Arrays.copyOf(tags, tags.length);
		final List<ForwardListener> fwdListenersCopy = new ArrayList<ForwardListener>();
		synchronized(fwdListeners){
			fwdListenersCopy.addAll(fwdListeners);
		}
		listenerExecutor.execute(()->{
			fwdListenersCopy.stream().forEach(
					f -> f.onForward(outputCopy, tagsCopy));
		});
	}
	
	public void addBackwardListener(BackwardListener listener){
		bwListeners.add(listener);
	}
	
	public void removeBackwardListener(BackwardListener listener){
		bwListeners.remove(listener);
	}
	
	protected void notifyBackwardListeners(){
		final Tensor gradInputCopy = gradInput.copyInto(null);
		final String[] tagsCopy = (tags == null) ? null : Arrays.copyOf(tags, tags.length);
		final List<BackwardListener> bwListenersCopy = new ArrayList<BackwardListener>();
		synchronized(bwListeners){
			bwListenersCopy.addAll(bwListeners);
		}
		listenerExecutor.execute(()->{
			bwListenersCopy.stream().forEach(
					b->b.onBackward(gradInputCopy, tagsCopy));
		});
	}
	
	protected final class ForwardRunnable implements Runnable {
		private final Module m;
		private final String[] tags;
		private final Tensor tensor;
		
		public ForwardRunnable(Module m, Tensor tensor, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = tensor;
		}
		
		public void run(){
			m.forward(id, tensor, tags);
			
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
		
		public BackwardRunnable(Module m, Tensor tensor, String[] tags){
			this.m = m;
			this.tags = tags;
			this.tensor = tensor;
		}
		
		public void run(){
			m.backward(id, tensor, tags);
		}
	}
	
}
